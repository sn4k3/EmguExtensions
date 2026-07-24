using System.Drawing;
using System.IO.Compression;
using Emgu.CV;
using Emgu.CV.CvEnum;

namespace EmguExtensions.Tests;

/// <summary>
/// Tests compressor-specific paths that are not exercised through <see cref="CMat"/>.
/// </summary>
public class UnitTestMatCompressor
{
    [Theory]
    [InlineData(32, CompressionLevel.NoCompression)]
    [InlineData(32, CompressionLevel.Fastest)]
    [InlineData(32, CompressionLevel.Optimal)]
    [InlineData(32, CompressionLevel.SmallestSize)]
    [InlineData(128, CompressionLevel.Optimal)]
    [InlineData(8192, CompressionLevel.Optimal)]
    public void Brotli_CompressContiguous_RoundTrips(int size, CompressionLevel compressionLevel)
    {
        using var source = CreatePatternMat(size, size);
        using var destination = new Mat(source.Size, source.Depth, source.NumberOfChannels);

        var compressed = MatCompressorBrotli.Instance.Compress(source, compressionLevel);
        MatCompressorBrotli.Instance.Decompress(compressed, destination);

        Assert.Equal(source.ToArray(), destination.ToArray());
    }

    [Fact]
    public void Brotli_CompressNonContiguous_RoundTrips()
    {
        using var source = CreatePatternMat(64, 48);
        using var roi = new Mat(source, new Rectangle(7, 5, 40, 30));
        using var destination = new Mat(roi.Size, roi.Depth, roi.NumberOfChannels);
        Assert.False(roi.IsContinuous);

        var compressed = MatCompressorBrotli.Instance.Compress(roi, CompressionLevel.Optimal);
        MatCompressorBrotli.Instance.Decompress(compressed, destination);

        Assert.Equal(roi.ToArray(), destination.ToArray());
    }

    private static Mat CreatePatternMat(int width, int height)
    {
        var mat = new Mat(height, width, DepthType.Cv8U, 1);
        var pixels = mat.GetSpan<byte>();
        for (var i = 0; i < pixels.Length; i++)
        {
            pixels[i] = (byte)((i * 31 + i / 17) % 256);
        }

        return mat;
    }
}
