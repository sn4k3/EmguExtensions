using System.Drawing;
using System.IO.Compression;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace EmguExtensions.Tests;

/// <summary>
/// Tests for <see cref="MatCompressorDelta"/>: byte-exact round trips for every depth/channel layout (random bits
/// exercise the wrapping arithmetic), ROI views, compression gain on smooth data, and wrapper validation.
/// </summary>
public class UnitTestMatCompressorDelta
{
    private static readonly DepthType[] AllDepths =
    [
        DepthType.Cv8U, DepthType.Cv8S, DepthType.Cv16U, DepthType.Cv16S, DepthType.Cv32S, DepthType.Cv32F,
        DepthType.Cv64F, DepthType.Cv16F, DepthType.Cv16BF, DepthType.CvBool, DepthType.Cv32U, DepthType.Cv64U,
        DepthType.Cv64S, DepthType.Cv8FE4M3FN, DepthType.Cv8FE4M3FNUZ,
    ];

    private static MatCompressorDelta[] DeltaCompressors =>
        MatCompressor.AvailableCompressors.OfType<MatCompressorDelta>().ToArray();

    public static TheoryData<string, DepthType, int> CompressorDepthChannels()
    {
        var data = new TheoryData<string, DepthType, int>();
        foreach (var compressor in DeltaCompressors)
        foreach (var depth in AllDepths)
        foreach (var channels in new[] { 1, 3, 4, 7 })
            data.Add(compressor.Name, depth, channels);
        return data;
    }

    private static Mat CreateRandomMat(DepthType depth, int channels, Size size, int seed = 1)
    {
        var mat = new Mat(size, depth, channels);
        new Random(seed).NextBytes(mat.GetSpanOfBytes());
        return mat;
    }

    private static Mat CreateSmoothMat(DepthType depth, Size size)
    {
        var mat = new Mat(size, depth, 1);
        using var gradient = new Mat(size, DepthType.Cv32F, 1);
        var values = new float[size.Width * size.Height];
        for (var y = 0; y < size.Height; y++)
        for (var x = 0; x < size.Width; x++)
            values[y * size.Width + x] = (float)(0.5 + 0.4 * Math.Sin(x / 40.0) * Math.Cos(y / 30.0));
        gradient.SetTo(values);
        gradient.ConvertTo(mat, depth, depth == DepthType.Cv16U ? ushort.MaxValue : 1);
        return mat;
    }

    private static void AssertBytesEqual(Mat expected, Mat actual)
    {
        Assert.Equal(expected.Size, actual.Size);
        Assert.Equal(expected.Depth, actual.Depth);
        Assert.Equal(expected.NumberOfChannels, actual.NumberOfChannels);
        for (var y = 0; y < expected.Height; y++)
            Assert.True(expected.GetReadOnlyRowSpanOfBytes(y).SequenceEqual(actual.GetReadOnlyRowSpanOfBytes(y)), $"Row {y} differs");
    }

    [Theory]
    [MemberData(nameof(CompressorDepthChannels))]
    public void RoundTrip_AnyDepthAndChannels_IsByteExact(string compressorName, DepthType depth, int channels)
    {
        var compressor = MatCompressor.GetCompressorByName(compressorName)!;
        using var mat = CreateRandomMat(depth, channels, new Size(37, 11));

        var compressed = compressor.Compress(mat, CompressionLevel.Fastest);
        using var dst = new Mat(mat.Size, depth, channels);
        compressor.Decompress(compressed, dst);

        AssertBytesEqual(mat, dst);
    }

    public static TheoryData<DepthType, int> WideRowLayouts()
    {
        // Every element size with 1-9 channels: channel periods below, equal to and above the 16-byte vector block
        var data = new TheoryData<DepthType, int>();
        foreach (var depth in new[] { DepthType.Cv8U, DepthType.Cv16U, DepthType.Cv32F, DepthType.Cv64F })
        for (var channels = 1; channels <= 9; channels++)
            data.Add(depth, channels);
        return data;
    }

    // Wide rows (many vector blocks plus an odd tail) with random and smooth data, checked against the stored bytes.
    [Theory]
    [MemberData(nameof(WideRowLayouts))]
    public void RoundTrip_WideRows_IsByteExact(DepthType depth, int channels)
    {
        using var mat = CreateRandomMat(depth, channels, new Size(203, 6), seed: channels);

        var compressed = MatCompressorDelta.Deflate.Compress(mat, CompressionLevel.Fastest);
        using var dst = new Mat(mat.Size, depth, channels);
        MatCompressorDelta.Deflate.Decompress(compressed, dst);

        AssertBytesEqual(mat, dst);
    }

    // Each value of the stored stream must be the wrapped difference from the previous same-channel element, so the
    // vectorized encoder/decoder can never drift from the documented storage format.
    [Fact]
    public void Compress_StoredFormat_IsPerChannelDeltaThenBytePlanes()
    {
        using var mat = new Mat(new Size(40, 1), DepthType.Cv16U, 3);
        var values = MemoryMarshalCast(mat);
        for (var i = 0; i < values.Length; i++)
            values[i] = (ushort)(i * 977 + 13);

        var stored = MatCompressorNone.Instance.Compress(mat); // Same bytes the delta wrapper would hand to Inner
        var filtered = new MatCompressorDelta(MatCompressorNone.Instance).Compress(mat);

        var count = values.Length;
        for (var e = 0; e < count; e++)
        {
            var expected = e < 3 ? values[e] : (ushort)(values[e] - values[e - 3]);
            var actual = (ushort)(filtered[e] | filtered[count + e] << 8);
            Assert.Equal(expected, actual);
        }
        Assert.Equal(stored.Length, filtered.Length);
    }

    private static Span<ushort> MemoryMarshalCast(Mat mat) =>
        System.Runtime.InteropServices.MemoryMarshal.Cast<byte, ushort>(mat.GetSpanOfBytes());

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public void RoundTrip_NarrowRows_IsByteExact(int width)
    {
        using var mat = CreateRandomMat(DepthType.Cv16U, 3, new Size(width, 5));

        var compressed = MatCompressorDelta.Deflate.Compress(mat);
        using var dst = new Mat(mat.Size, mat.Depth, mat.NumberOfChannels);
        MatCompressorDelta.Deflate.Decompress(compressed, dst);

        AssertBytesEqual(mat, dst);
    }

    [Fact]
    public void RoundTrip_RoiViews_SourceAndDestination_IsByteExact()
    {
        using var source = CreateRandomMat(DepthType.Cv32F, 3, new Size(64, 40));
        using var sourceView = new Mat(source, new Rectangle(5, 3, 40, 20));
        using var destination = new Mat(new Size(64, 40), DepthType.Cv32F, 3);
        destination.SetTo(new MCvScalar(0));
        using var destinationView = new Mat(destination, new Rectangle(10, 12, 40, 20));

        var compressed = MatCompressorDelta.Brotli.Compress(sourceView);
        MatCompressorDelta.Brotli.Decompress(compressed, destinationView);

        AssertBytesEqual(sourceView, destinationView);
        Assert.Equal([0, 0, 0, 0], destination.GetRawData(0, 0)[..4]); // outside the view is untouched
    }

    [Fact]
    public void CMat_WithDeltaCompressor_RoundTripsAndRecordsDecompressor()
    {
        using var mat = CreateSmoothMat(DepthType.Cv16U, new Size(256, 128));

        var cmat = new CMat(mat, MatCompressorDelta.Deflate);
        using var decompressed = cmat.Decompress();

        Assert.Same(MatCompressorDelta.Deflate, cmat.Decompressor);
        AssertBytesEqual(mat, decompressed);
    }

    [Theory]
    [InlineData(DepthType.Cv16U)]
    [InlineData(DepthType.Cv32F)]
    public void Compress_SmoothData_IsMuchSmallerThanInnerCompressor(DepthType depth)
    {
        using var mat = CreateSmoothMat(depth, new Size(512, 256));

        var plain = MatCompressorDeflate.Instance.Compress(mat, CompressionLevel.Optimal);
        var delta = MatCompressorDelta.Deflate.Compress(mat, CompressionLevel.Optimal);

        Assert.True(delta.Length * 1.3 < plain.Length, $"Delta {delta.Length} should be well below plain {plain.Length}");
    }

    [Fact]
    public void Identity_DerivesFromInnerCompressor()
    {
        var compressor = MatCompressorDelta.Brotli;

        Assert.Same(MatCompressorBrotli.Instance, compressor.Inner);
        Assert.Equal("Brotli+Delta", compressor.Name);
        Assert.Equal(MatCompressorBrotli.Instance.Provider, compressor.Provider);
        Assert.Equal(MatCompressorBrotli.Instance.MaximumCompressionLevel, compressor.MaximumCompressionLevel);
        Assert.Same(compressor, MatCompressor.GetCompressorById(compressor.Id));
        Assert.NotEqual<MatCompressor>(MatCompressorBrotli.Instance, compressor);
    }

    [Fact]
    public void Compress_InvalidNativeLevel_Throws()
    {
        using var mat = CreateRandomMat(DepthType.Cv8U, 1, new Size(8, 8));

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            MatCompressorDelta.Brotli.Compress(mat, MatCompressorBrotli.Instance.MaximumCompressionLevel + 1));
    }

    [Fact]
    public void Constructor_ImageCodecOrDeltaInner_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new MatCompressorDelta(null!));
        Assert.Throws<ArgumentException>(() => new MatCompressorDelta(MatCompressorPng.Instance));
        Assert.Throws<ArgumentException>(() => new MatCompressorDelta(MatCompressorTiff.Instance));
        Assert.Throws<ArgumentException>(() => new MatCompressorDelta(MatCompressorDelta.Deflate));
    }

    [Fact]
    public void Constructor_CustomInner_RoundTrips()
    {
        var compressor = new MatCompressorDelta(MatCompressorGZip.Instance);
        using var mat = CreateRandomMat(DepthType.Cv64F, 2, new Size(20, 9));

        var compressed = compressor.Compress(mat);
        using var dst = new Mat(mat.Size, mat.Depth, mat.NumberOfChannels);
        compressor.Decompress(compressed, dst);

        Assert.Equal("GZip+Delta", compressor.Name);
        AssertBytesEqual(mat, dst);
    }
}
