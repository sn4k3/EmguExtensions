using System.Drawing;
using System.IO.Compression;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Order;
using Emgu.CV;
using Emgu.CV.CvEnum;

namespace EmguExtensions.Benchmarks;

/// <summary>
/// Benchmarks compress and decompress for every registered <see cref="MatCompressor"/>
/// at every <see cref="CompressionLevel"/>.
/// </summary>
[MemoryDiagnoser]
[Orderer(SummaryOrderPolicy.FastestToSlowest)]
[GroupBenchmarksBy(BenchmarkDotNet.Configs.BenchmarkLogicalGroupRule.ByCategory)]
[MaxIterationCount(16)]
public class MatCompressorBenchmarks
{
    /// <summary>
    /// Mat sizes used in the benchmarks (width x height).
    /// FullHD, 4K, 8K. These sizes are chosen to represent common resolutions and to provide a range of data sizes for testing the performance of the compressors.
    /// </summary>
    [Params(1920 /*/ 2/*, 3840 * 2160 / 2, 7680 * 4320 / 2*/)]
    public int MatSize { get; set; }

    [ParamsSource(nameof(CompressorNames))]
    public string CompressorName { get; set; } = null!;

    [ParamsAllValues]
    public CompressionLevel Level { get; set; }

    private Mat _mat = null!;
    private Mat _decompressDst = null!;
    private MatCompressor _compressor = null!;
    private byte[] _compressedBytes = null!;

    /// <summary>
    /// Provides the compressor names from the registered collection.
    /// </summary>
    public IEnumerable<string> CompressorNames()
        => MatCompressor.AvailableCompressors.Select(c => c.Name);

    [GlobalSetup]
    public void Setup()
    {
        _compressor = MatCompressor.GetCompressorByName(CompressorName)
            ?? throw new InvalidOperationException($"Unknown compressor: {CompressorName}");

        // Create a realistic test image: gradient + white rectangle (not trivially compressible).
        _mat = new Mat(MatSize, MatSize, DepthType.Cv8U, 1);
        var span = _mat.GetSpan<byte>();
        for (int i = 0; i < span.Length; i++)
            span[i] = (byte)(i % 256);
        CvInvoke.Rectangle(_mat,
            new Rectangle(MatSize / 4, MatSize / 4, MatSize / 2, MatSize / 2),
            EmguExtensions.WhiteColor, -1);

        // Pre-compress for decompress benchmark.
        _compressedBytes = _compressor.Compress(_mat, Level);
        _decompressDst = new Mat(MatSize, MatSize, DepthType.Cv8U, 1);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _mat.Dispose();
        _decompressDst.Dispose();
    }

    [Benchmark(Description = "Compress")]
    [BenchmarkCategory("Compress")]
    public byte[] Compress()
    {
        return _compressor.Compress(_mat, Level);
    }

    [Benchmark(Description = "Decompress")]
    [BenchmarkCategory("Decompress")]
    public byte[] Decompress()
    {
        _compressor.Decompress(_compressedBytes, _decompressDst);
        return _compressedBytes;
    }
}
