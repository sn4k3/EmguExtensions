#pragma warning disable xUnit1051

using System.Drawing;
using System.IO.Compression;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Xunit;

namespace EmguExtensions.Tests;

/// <summary>
/// Regression tests for the fixes made after the MatCompressor / Extensions review.
/// </summary>
public class UnitTestReviewFixes
{
    /// <summary>
    /// Compressor with a scripted compress behavior, to exercise the <see cref="CMat"/> fallbacks.
    /// </summary>
    private sealed class ScriptedCompressor(Func<Mat, byte[]> compress) : MatCompressor
    {
        public override string Name => "Scripted";

        protected override byte[] CompressCore(Mat src, int compressionLevel) => compress(src);

        protected override void DecompressCore(byte[] compressedBytes, Mat dst) =>
            throw new NotSupportedException();
    }

    private static Mat CreatePatternMat(int width = 64, int height = 48, int channels = 1)
    {
        var mat = new Mat(new Size(width, height), DepthType.Cv8U, channels);
        var span = mat.GetSpan<byte>();
        for (var i = 0; i < span.Length; i++)
        {
            span[i] = (byte)(i / 64 % 2 == 0 ? 0 : 200); // Compressible
        }

        return mat;
    }

    #region MatCompressor

    [Theory]
    [InlineData("Brotli")]
    [InlineData("Deflate")]
    [InlineData("GZip")]
    [InlineData("ZLib")]
    [InlineData("None")]
    public void Decompress_EmptyDestination_ThrowsForCompressorsThatNeedAllocatedDestination(string name)
    {
        var compressor = MatCompressor.GetCompressorByName(name)!;
        using var source = CreatePatternMat();
        var compressed = compressor.Compress(source);

        using var empty = new Mat();

        Assert.Throws<ArgumentException>(() => compressor.Decompress(compressed, empty));
    }

    [Fact]
    public async Task DecompressAsync_EmptyDestination_ThrowsForCompressorsThatNeedAllocatedDestination()
    {
        using var source = CreatePatternMat();
        var compressed = MatCompressorGZip.Instance.Compress(source);

        using var empty = new Mat();

        await Assert.ThrowsAsync<ArgumentException>(() =>
            MatCompressorGZip.Instance.DecompressAsync(compressed, empty)
        );
    }

    [Fact]
    public void Png_Decompress_EmptyDestination_AllocatesDestination()
    {
        using var source = CreatePatternMat(channels: 3);
        var compressed = MatCompressorPng.Instance.Compress(source);

        using var dst = new Mat();
        MatCompressorPng.Instance.Decompress(compressed, dst);

        Assert.Equal(source.Size, dst.Size);
        Assert.Equal(source.ToArray(), dst.ToArray());
    }

    [Fact]
    public void Png_Decompress_CorruptData_Throws()
    {
        using var dst = new Mat(new Size(8, 8), DepthType.Cv8U, 1);

        Assert.Throws<InvalidDataException>(() =>
            MatCompressorPng.Instance.Decompress([1, 2, 3, 4, 5, 6, 7, 8], dst)
        );
    }

    [Fact]
    public void None_Decompress_LengthMismatch_Throws()
    {
        using var dst = new Mat(new Size(8, 8), DepthType.Cv8U, 1);

        Assert.Throws<InvalidDataException>(() =>
            MatCompressorNone.Instance.Decompress(new byte[10], dst)
        );
    }

    [Fact]
    public void None_Decompress_CopiesIntoExactDestination()
    {
        using var dst = new Mat(new Size(4, 4), DepthType.Cv8U, 1);
        var raw = Enumerable.Range(0, 16).Select(i => (byte)(i + 1)).ToArray();

        MatCompressorNone.Instance.Decompress(raw, dst);

        Assert.Equal(raw, dst.ToArray());
    }

    [Fact]
    public void Id_IsStableAndCombinesProviderAndName()
    {
        Assert.Equal(".NET#Brotli", MatCompressorBrotli.Instance.Id);
        Assert.Same(MatCompressorBrotli.Instance.Id, MatCompressorBrotli.Instance.Id);
        Assert.Same(MatCompressorBrotli.Instance, MatCompressor.GetCompressorById(".net#brotli"));
    }

    #endregion

    #region CMat

    [Fact]
    public void CMat_Compress_CompressorReturnsEmptyArray_StoresRawInsteadOfLosingData()
    {
        using var mat = CreatePatternMat();
        var cmat = new CMat(new ScriptedCompressor(_ => []), mat.Width, mat.Height);

        cmat.Compress(mat);

        Assert.False(cmat.IsEmpty);
        Assert.False(cmat.IsCompressed);
        using var back = cmat.Decompress();
        Assert.Equal(mat.ToArray(), back.ToArray());
    }

    [Fact]
    public void CMat_Compress_CompressorThrowsCodecError_FallsBackToRaw()
    {
        using var mat = CreatePatternMat();
        var cmat = new CMat(
            new ScriptedCompressor(_ => throw new InvalidDataException("codec failure")),
            mat.Width,
            mat.Height
        );

        cmat.Compress(mat);

        Assert.False(cmat.IsCompressed);
        using var back = cmat.Decompress();
        Assert.Equal(mat.ToArray(), back.ToArray());
    }

    [Fact]
    public void CMat_Compress_CompressorThrowsArgumentException_Propagates()
    {
        using var mat = CreatePatternMat();
        var cmat = new CMat(
            new ScriptedCompressor(_ => throw new ArgumentOutOfRangeException("level")),
            mat.Width,
            mat.Height
        );

        Assert.Throws<ArgumentOutOfRangeException>(() => cmat.Compress(mat));
    }

    [Fact]
    public void CMat_Compress_NoneCompressor_StoresRaw()
    {
        using var mat = CreatePatternMat();

        var cmat = new CMat(mat, MatCompressorNone.Instance);

        Assert.False(cmat.IsCompressed);
        Assert.Same(MatCompressorNone.Instance, cmat.Decompressor);
        Assert.Equal(mat.ToArray(), cmat.CompressedBytes);
    }

    [Fact]
    public void CMat_Constructor_NullCompressionLevel_UsesDefaultCompressionLevel()
    {
        using var mat = CreatePatternMat();

        var cmat = new CMat(mat, MatCompressorGZip.Instance);

        Assert.Equal(MatCompressor.DefaultCompressionLevel, cmat.CompressionLevel);
    }

    [Fact]
    public void CMat_SetCompressedBytes_ClearsStaleRoi()
    {
        using var mat = CreatePatternMat(100, 80);
        using var matRoi = new MatRoi(mat, new Rectangle(10, 10, 30, 30));
        var cmat = new CMat(matRoi, MatCompressorGZip.Instance);
        Assert.Equal(new Rectangle(10, 10, 30, 30), cmat.Roi);

        var raw = new byte[100 * 80];
        cmat.SetCompressedBytes(raw, MatCompressorNone.Instance);

        Assert.Equal(Rectangle.Empty, cmat.Roi);
        using var decompressed = cmat.Decompress();
        Assert.Equal(new Size(100, 80), decompressed.Size);
    }

    [Fact]
    public void CMat_Decompress_BytesDoNotMatchDescription_Throws()
    {
        var cmat = new CMat(10, 10);
        cmat.SetCompressedBytes(new byte[40], MatCompressorNone.Instance);

        Assert.Throws<InvalidDataException>(() => cmat.Decompress());
    }

    [Fact]
    public void CMat_ChangeCompressor_ReEncode_RoundTrips()
    {
        using var mat = CreatePatternMat();
        var cmat = new CMat(mat, MatCompressorGZip.Instance);
        Assert.True(cmat.IsCompressed);

        var changed = cmat.ChangeCompressor(MatCompressorBrotli.Instance, reEncodeWithNewCompressor: true);

        Assert.True(changed);
        Assert.Same(MatCompressorBrotli.Instance, cmat.Decompressor);
        using var back = cmat.Decompress();
        Assert.Equal(mat.ToArray(), back.ToArray());
    }

    [Fact]
    public async Task CMat_ConcurrentReadsAndWrites_AreConsistent()
    {
        using var matA = CreatePatternMat(64, 48);
        using var matB = CreatePatternMat(128, 96);
        var shared = new CMat(matA, MatCompressorGZip.Instance);
        var other = new CMat(matA, MatCompressorGZip.Instance);
        var errors = new List<Exception>();

        void Run(Action action)
        {
            try
            {
                for (var i = 0; i < 200; i++)
                    action();
            }
            catch (Exception ex)
            {
                lock (errors)
                    errors.Add(ex);
            }
        }

        var tasks = new[]
        {
            Task.Run(() => Run(() => shared.Compress(matB))),
            Task.Run(() => Run(() => shared.Compress(matA))),
            Task.Run(() => Run(() => _ = shared.Equals(other))),
            Task.Run(() => Run(() => _ = shared.GetHashCode())),
            Task.Run(() => Run(() => _ = shared.ToString())),
            Task.Run(() =>
                Run(() =>
                {
                    // Every statistic must come from one consistent snapshot, whatever the size currently is
                    var ratio = shared.CompressionRatio;
                    var saved = shared.SavedBytes;
                    Assert.True(ratio >= 0);
                    Assert.True(saved >= 0);
                })
            ),
            Task.Run(() =>
                Run(() =>
                {
                    using var decompressed = shared.Decompress();
                    Assert.True(
                        decompressed.Size == matA.Size || decompressed.Size == matB.Size,
                        $"Unexpected size {decompressed.Size}"
                    );
                })
            ),
        };

        await Task.WhenAll(tasks);

        Assert.Empty(errors);
    }

    [Fact]
    public void CMat_ThresholdAndCompressorSetters_AreApplied()
    {
        using var mat = CreatePatternMat();
        var cmat = new CMat { ThresholdToCompress = int.MaxValue };

        cmat.Compress(mat);

        Assert.False(cmat.IsCompressed); // Below the threshold

        cmat.ThresholdToCompress = 0;
        cmat.Compressor = MatCompressorGZip.Instance;
        cmat.CompressionLevel = CompressionLevel.SmallestSize;
        cmat.Compress(mat);

        Assert.True(cmat.IsCompressed);
        Assert.Throws<ArgumentNullException>(() => cmat.Compressor = null!);
    }

    #endregion

    #region Extensions

    [Fact]
    public void ShrinkToFitPreserveAspect_ThinImage_NeverCollapsesADimension()
    {
        using var thin = new Mat(new Size(10000, 1), DepthType.Cv8U, 1);

        var shrunk = thin.ShrinkToFitPreserveAspect(100, 100);

        Assert.True(shrunk);
        Assert.Equal(new Size(100, 1), thin.Size);
    }

    [Fact]
    public void ShrinkToFitPreserveAspect_InvalidMaximum_Throws()
    {
        using var mat = new Mat(new Size(10, 10), DepthType.Cv8U, 1);

        Assert.Throws<ArgumentOutOfRangeException>(() => mat.ShrinkToFitPreserveAspect(0, 10));
        Assert.Throws<ArgumentOutOfRangeException>(() => mat.ShrinkToFitPreserveAspect(10, -1));
    }

    [Fact]
    public void Resize_TinyScale_ClampsToOnePixel()
    {
        using var mat = new Mat(new Size(100, 10), DepthType.Cv8U, 1);

        mat.Resize(0.01);

        Assert.Equal(new Size(1, 1), mat.Size);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    [InlineData(double.NaN)]
    public void Resize_InvalidScale_Throws(double scale)
    {
        using var mat = new Mat(new Size(10, 10), DepthType.Cv8U, 1);

        Assert.Throws<ArgumentOutOfRangeException>(() => mat.Resize(scale));
    }

    [Fact]
    public void Roi_ZeroWidthAtNonZeroOrigin_IsTreatedAsEmpty()
    {
        using var mat = new Mat(new Size(20, 10), DepthType.Cv8U, 1);

        using var captured = mat.Roi(new Rectangle(5, 0, 0, 10), EmptyRoiBehavior.CaptureSource);

        Assert.Equal(mat.Size, captured.Size);
        Assert.Throws<InvalidOperationException>(() =>
            mat.Roi(new Rectangle(5, 0, 0, 10), EmptyRoiBehavior.ThrowException)
        );
    }

    [Fact]
    public void GetMemory2D_NegativeRoi_ThrowsLikeSpan2D()
    {
        using var mat = new Mat(new Size(20, 10), DepthType.Cv8U, 1);
        var roi = new Rectangle(0, 0, -5, 5);

        Assert.Throws<ArgumentOutOfRangeException>(() => mat.GetSpan2DOfBytes(roi));
        Assert.Throws<ArgumentOutOfRangeException>(() => mat.GetMemory2DOfBytes(roi));
        Assert.Throws<ArgumentOutOfRangeException>(() => mat.GetReadOnlyMemory2DOfBytes(roi));
    }

    [Fact]
    public void CreateMask_ColorSource_ReturnsSingleChannelMask()
    {
        using var color = new Mat(new Size(20, 20), DepthType.Cv8U, 3);
        Point[][] contours = [[new(2, 2), new(15, 2), new(15, 15), new(2, 15)]];

        using var mask = color.CreateMask(contours);

        Assert.Equal(1, mask.NumberOfChannels);
        Assert.Equal(DepthType.Cv8U, mask.Depth);
        Assert.Equal(color.Size, mask.Size);
        Assert.True(mask.CountNonZero > 0);
    }

    [Fact]
    public void FactorColor_RoundsInsteadOfTruncating()
    {
        var color = Color.FromArgb(255, 100, 100, 100);

        var factored = color.FactorColor(0.555); // 55.5 -> 56

        Assert.Equal(56, factored.R);
        Assert.Equal(56, factored.G);
        Assert.Equal(56, factored.B);
    }

    [Fact]
    public void GetBitmapInfo_RoiMat_ReportsTheRealStride()
    {
        using var big = new Mat(new Size(40, 30), DepthType.Cv8U, 3);
        using var roi = new Mat(big, new Rectangle(5, 4, 10, 8));
        Assert.False(roi.IsContinuous);

        var info = roi.GetBitmapInfo();

        Assert.Equal(big.Step, info.RowBytes); // The pitch to the next row, not the 30 bytes of row data
        Assert.False(info.IsContiguous);
        Assert.Equal(3, info.BytesPerPixel);
        Assert.Equal(new Size(10, 8), info.Size);

        using var continuous = roi.Clone();
        Assert.Equal(30, continuous.GetBitmapInfo().RowBytes);
        Assert.True(continuous.GetBitmapInfo().IsContiguous);
    }

    [Fact]
    public void FindLength_PointF_ReturnsEuclideanDistance()
    {
        Assert.Equal(5.0, PointExtensions.FindLength(new PointF(0, 0), new PointF(3, 4)), 6);
    }

    [Fact]
    public void RotateAdjustBounds_RoundsBoundsUp()
    {
        using var mat = new Mat(new Size(100, 100), DepthType.Cv8U, 1);

        mat.RotateAdjustBounds(45);

        // 100 * (sin45 + cos45) = 141.42..., it must not be cropped to 141
        Assert.Equal(new Size(142, 142), mat.Size);
    }

    [Fact]
    public void ScanLines_Vertical_OrdersByColumnThenRow_WithMultipleRunsPerColumn()
    {
        using var mat = new Mat(new Size(3, 6), DepthType.Cv8U, 1);
        mat.SetTo(new MCvScalar(0));
        // Column 0: runs rows 0-1 (255) and 3-4 (255), column 2: rows 1-4 (255 then 100 at 3-4)
        foreach (var y in new[] { 0, 1, 3, 4 })
            mat.SetByte(0, y, 255);
        foreach (var y in new[] { 1, 2 })
            mat.SetByte(2, y, 255);
        foreach (var y in new[] { 3, 4 })
            mat.SetByte(2, y, 100);

        var lines = mat.ScanLines(vertically: true, offset: new Point(10, 20));

        Assert.Equal(
            [
                new GreyLine { StartX = 10, StartY = 20, EndX = 10, EndY = 21, Grey = 255 },
                new GreyLine { StartX = 10, StartY = 23, EndX = 10, EndY = 24, Grey = 255 },
                new GreyLine { StartX = 12, StartY = 21, EndX = 12, EndY = 22, Grey = 255 },
                new GreyLine { StartX = 12, StartY = 23, EndX = 12, EndY = 24, Grey = 100 },
            ],
            lines
        );
    }

    [Fact]
    public void ScanLines_Vertical_NonContinuousRoi_MatchesContinuousClone()
    {
        using var big = new Mat(new Size(30, 20), DepthType.Cv8U, 1);
        var span = big.GetSpan<byte>();
        for (var i = 0; i < span.Length; i++)
            span[i] = (byte)(i % 7 < 3 ? 0 : 255);

        using var roi = new Mat(big, new Rectangle(5, 3, 12, 9));
        using var clone = roi.Clone();
        Assert.False(roi.IsContinuous);

        Assert.Equal(clone.ScanLines(vertically: true), roi.ScanLines(vertically: true));
        Assert.Equal(
            clone.ScanLines(v => v, vertically: true),
            roi.ScanLines(v => v, vertically: true)
        );
    }

    [Fact]
    public void GetSvgPath_IsCultureIndependent()
    {
        using var mat = new Mat(new Size(20, 20), DepthType.Cv8U, 1);
        mat.SetTo(new MCvScalar(0));
        CvInvoke.Rectangle(mat, new Rectangle(4, 4, 10, 10), new MCvScalar(255), -1);

        var original = System.Globalization.CultureInfo.CurrentCulture;
        try
        {
            System.Globalization.CultureInfo.CurrentCulture = new System.Globalization.CultureInfo("ar-SA");
            var paths = mat.GetSvgPath();

            Assert.Single(paths);
            Assert.StartsWith("M ", paths[0]);
            Assert.All(paths[0], c => Assert.True(c < 128, $"Unexpected non-ASCII char '{c}'"));
        }
        finally
        {
            System.Globalization.CultureInfo.CurrentCulture = original;
        }
    }

    [Fact]
    public void CopyAreas_NestedIslandInsideHole_IsAddedToTheGroupArea()
    {
        // Ring (outer 100x100, hole 60x60) with an island (30x30) inside the hole:
        // net area is roughly 10000 - 3600 + 900 = 7300, not the 10000 - 3600 - 900 = 5500 of a plain subtraction
        using var src = new Mat(new Size(200, 120), DepthType.Cv8U, 1);
        src.SetTo(new MCvScalar(0));
        CvInvoke.Rectangle(src, new Rectangle(10, 10, 100, 100), new MCvScalar(255), -1);
        CvInvoke.Rectangle(src, new Rectangle(30, 30, 60, 60), new MCvScalar(0), -1);
        CvInvoke.Rectangle(src, new Rectangle(45, 45, 30, 30), new MCvScalar(255), -1);

        using var larger = src.NewZeros();
        src.CopyAreasLargerThan(6500, larger);
        Assert.Equal(255, larger.GetByte(15, 15)); // Ring copied
        Assert.Equal(255, larger.GetByte(60, 60)); // Island copied with its group
        Assert.Equal(0, larger.GetByte(35, 35)); // Hole left alone

        using var smaller = src.NewZeros();
        src.CopyAreasSmallerThan(6500, smaller);
        Assert.Equal(0, smaller.GetByte(15, 15));
        Assert.Equal(0, smaller.GetByte(60, 60));
    }

    [Fact]
    public void PutTextRotated_Rotated90_RunsAlongTheOtherAxis()
    {
        using var horizontal = new Mat(new Size(300, 300), DepthType.Cv8U, 1);
        horizontal.SetTo(new MCvScalar(0));
        using var vertical = horizontal.Clone();
        var org = new Point(150, 150);

        horizontal.PutTextRotated(
            "Hello",
            org,
            FontFace.HersheySimplex,
            1.5,
            new MCvScalar(255),
            2,
            lineGapOffset: 0
        );
        vertical.PutTextRotated(
            "Hello",
            org,
            FontFace.HersheySimplex,
            1.5,
            new MCvScalar(255),
            2,
            lineGapOffset: 0,
            angle: 90
        );

        var horizontalBox = CvInvoke.BoundingRectangle(horizontal);
        var verticalBox = CvInvoke.BoundingRectangle(vertical);

        Assert.True(horizontalBox.Width > horizontalBox.Height);
        Assert.True(verticalBox.Height > verticalBox.Width);
        // The text origin is the pivot: both boxes touch the neighborhood of it
        Assert.True(Rectangle.Inflate(horizontalBox, 10, 10).Contains(org));
        Assert.True(Rectangle.Inflate(verticalBox, 10, 10).Contains(org));
        // Same glyphs, same amount of ink (rotating by 90 degrees is lossless)
        Assert.InRange(vertical.CountNonZero, horizontal.CountNonZero - 40, horizontal.CountNonZero + 40);
    }

    [Fact]
    public void PutTextRotated_AnyAngle_OnlyTouchesTheSurroundingsOfTheText()
    {
        using var mat = new Mat(new Size(400, 400), DepthType.Cv8U, 1);
        mat.SetTo(new MCvScalar(0));

        mat.PutTextRotated(
            "Rotated text",
            new Point(200, 200),
            FontFace.HersheySimplex,
            1,
            new MCvScalar(255),
            2,
            lineGapOffset: 0,
            angle: 33
        );

        var box = CvInvoke.BoundingRectangle(mat);
        Assert.True(mat.CountNonZero > 0);
        Assert.True(box.Width < 250 && box.Height < 250);
        Assert.True(Rectangle.Inflate(box, 10, 10).Contains(new Point(200, 200)));
    }

    [Fact]
    public void PutTextRotated_PartiallyOutsideAndFullyOutside_AreHandled()
    {
        using var mat = new Mat(new Size(100, 100), DepthType.Cv8U, 1);
        mat.SetTo(new MCvScalar(0));

        mat.PutTextRotated("Clipped text", new Point(-20, 50), FontFace.HersheySimplex, 1, new MCvScalar(255), 2,
            lineGapOffset: 0, angle: 20);
        Assert.True(mat.CountNonZero > 0);

        using var untouched = new Mat(new Size(100, 100), DepthType.Cv8U, 1);
        untouched.SetTo(new MCvScalar(0));
        untouched.PutTextRotated("Far away", new Point(5000, 5000), FontFace.HersheySimplex, 1, new MCvScalar(255), 2,
            lineGapOffset: 0, angle: 20);
        Assert.Equal(0, untouched.CountNonZero);
    }

    #endregion
}
