#pragma warning disable xUnit1051

using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Xunit;

namespace EmguExtensions.Tests;

/// <summary>
/// Unit tests covering fixes and improvements for <see cref="MatCompressor"/>,
/// <see cref="EmguContours"/>, <see cref="EmguContour"/>, and <see cref="EmguContourFamily"/>.
/// </summary>
public class UnitTestCompressorAndContourFixes
{
    [Theory]
    [InlineData("Brotli")]
    [InlineData("Deflate")]
    [InlineData("GZip")]
    [InlineData("ZLib")]
    [InlineData("None")]
    [InlineData("PNG")]
    public void Decompress_IntoNonContinuousRoi_ProducesExactMatch(string compressorName)
    {
        var compressor = MatCompressor.GetCompressorByName(compressorName)!;
        Assert.NotNull(compressor);

        // Create a source pattern
        using var source = new Mat(30, 40, DepthType.Cv8U, 1);
        var srcSpan = source.GetSpan<byte>();
        for (var i = 0; i < srcSpan.Length; i++)
        {
            srcSpan[i] = (byte)((i * 37 + 13) % 256);
        }

        var compressed = compressor.Compress(source);

        // Destination is a non-continuous sub-matrix ROI within a larger Mat
        using var fullDst = new Mat(60, 80, DepthType.Cv8U, 1);
        fullDst.SetTo(new MCvScalar(0));

        using var roiDst = new Mat(fullDst, new Rectangle(10, 10, 40, 30));
        Assert.False(roiDst.IsContinuous);

        compressor.Decompress(compressed, roiDst);

        Assert.Equal(source.ToArray(), roiDst.ToArray());
    }

    [Fact]
    public void MatCompressor_DefaultCompressor_ThrowsOnNull()
    {
        Assert.Throws<ArgumentNullException>(() => MatCompressor.DefaultCompressor = null!);
    }

    [Fact]
    public async Task MatCompressor_EmptyBytes_HandledSafely()
    {
        using var mat = new Mat(10, 10, DepthType.Cv8U, 1);
        var compressor = MatCompressorBrotli.Instance;

        // Synchronous Decompress with empty bytes should do nothing
        compressor.Decompress([], mat);

        // Asynchronous DecompressAsync with empty bytes should do nothing
        await compressor.DecompressAsync([], mat);
    }

    [Fact]
    public async Task MatCompressor_CompressAsync_ThrowsWhenCancelled()
    {
        using var mat = new Mat(10, 10, DepthType.Cv8U, 1);
        using var cts = new CancellationTokenSource();
        await cts.CancelAsync();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
        {
            await MatCompressorBrotli.Instance.CompressAsync(mat, cts.Token);
        });
    }

    [Fact]
    public void EmguContour_Empty_CentroidReturnsAnchorCenter()
    {
        using var emptyVec = new VectorOfPoint();
        using var contour = new EmguContour(emptyVec);

        Assert.True(contour.IsEmpty);
        Assert.Equal(EmguCvExtensions.AnchorCenter, contour.Centroid);
    }

    [Fact]
    public void EmguContour_IEquatable_WorksCorrectly()
    {
        var pts1 = new[] { new Point(0, 0), new Point(10, 0), new Point(10, 10), new Point(0, 10) };
        var pts2 = new[] { new Point(0, 0), new Point(10, 0), new Point(10, 10), new Point(0, 10) };
        var pts3 = new[] { new Point(0, 0), new Point(5, 0), new Point(5, 5), new Point(0, 5) };

        using var c1 = new EmguContour(pts1);
        using var c2 = new EmguContour(pts2);
        using var c3 = new EmguContour(pts3);

        Assert.True(c1.Equals(c2));
        Assert.True(c2.Equals(c1));
        Assert.False(c1.Equals(c3));
        Assert.False(c1.Equals((EmguContour?)null));
        Assert.Equal(c1.GetHashCode(), c2.GetHashCode());
    }

    [Fact]
    public void EmguContourFamily_Root_TraversesToTopParent()
    {
        var outerPts = new[] { new Point(0, 0), new Point(100, 0), new Point(100, 100), new Point(0, 100) };
        var holePts = new[] { new Point(10, 10), new Point(90, 10), new Point(90, 90), new Point(10, 90) };
        var innerPts = new[] { new Point(20, 20), new Point(80, 20), new Point(80, 80), new Point(20, 80) };

        using var vec = new VectorOfVectorOfPoint([new VectorOfPoint(outerPts), new VectorOfPoint(holePts), new VectorOfPoint(innerPts)]);
        // outer (0) -> hole (1) -> inner (2)
        var hierarchy = new int[,]
        {
            { -1, -1, 1, -1 },
            { -1, -1, 2, 0 },
            { -1, -1, -1, 1 }
        };

        using var contours = new EmguContours(vec, hierarchy);
        Assert.Single(contours.Families);

        var root = contours.Families[0];
        Assert.Single(root);
        var hole = root[0];
        Assert.Single(hole);
        var inner = hole[0];

        Assert.Same(root, root.Root);
        Assert.Same(root, hole.Root);
        Assert.Same(root, inner.Root);
        Assert.Equal(0, root.Depth);
        Assert.Equal(1, hole.Depth);
        Assert.Equal(2, inner.Depth);
    }

    [Fact]
    public void EmguContours_MinMaxSolidArea_HandlesEmptyGracefully()
    {
        using var emptyVec = new VectorOfVectorOfPoint();
        var emptyHierarchy = new int[0, 4];
        using var contours = new EmguContours(emptyVec, emptyHierarchy);

        Assert.True(contours.IsEmpty);
        Assert.Equal(0, contours.TotalSolidArea);
        Assert.Equal(0, contours.MinSolidArea);
        Assert.Equal(0, contours.MaxSolidArea);
    }

    [Fact]
    public void EmguContours_GetLargestContourArea_SkipsNonRootContours()
    {
        var outerPts = new[] { new Point(0, 0), new Point(100, 0), new Point(100, 100), new Point(0, 100) };
        var holePts = new[] { new Point(10, 10), new Point(90, 10), new Point(90, 90), new Point(10, 90) };

        using var vec = new VectorOfVectorOfPoint([new VectorOfPoint(outerPts), new VectorOfPoint(holePts)]);
        // Hierarchy where contour 0 has no parent (-1), and contour 1 has parent 0
        var hierarchy = new int[,]
        {
            { -1, -1, 1, -1 },
            { -1, -1, -1, 0 }
        };

        var largestExternal = EmguContours.GetLargestContourArea(vec, hierarchy);
        var largestOverall = EmguContours.GetLargestContourArea(vec);

        Assert.True(largestExternal > 0);
        Assert.True(largestOverall > 0);
        Assert.Equal(CvInvoke.ContourArea(vec[0]), largestExternal);
    }

    [Fact]
    public void EmguContours_ContoursIntersectingPixels_CalculatesCorrectly()
    {
        var box1 = new[] { new Point(0, 0), new Point(20, 0), new Point(20, 20), new Point(0, 20) };
        var box2 = new[] { new Point(10, 10), new Point(30, 10), new Point(30, 30), new Point(10, 30) };
        var box3 = new[] { new Point(100, 100), new Point(120, 100), new Point(120, 120), new Point(100, 120) };

        using var vec1 = new VectorOfVectorOfPoint([new VectorOfPoint(box1)]);
        using var vec2 = new VectorOfVectorOfPoint([new VectorOfPoint(box2)]);
        using var vec3 = new VectorOfVectorOfPoint([new VectorOfPoint(box3)]);

        // Intersecting boxes 1 and 2
        var pixels12 = EmguContours.ContoursIntersectingPixels(vec1, vec2);
        Assert.True(pixels12 > 0);
        Assert.True(EmguContours.ContoursIntersect(vec1, vec2));

        // Disjoint boxes 1 and 3
        var pixels13 = EmguContours.ContoursIntersectingPixels(vec1, vec3);
        Assert.Equal(0, pixels13);
        Assert.False(EmguContours.ContoursIntersect(vec1, vec3));
    }
}
