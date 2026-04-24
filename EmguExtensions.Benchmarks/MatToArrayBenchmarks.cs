using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Order;
using Emgu.CV;
using Emgu.CV.CvEnum;

namespace EmguExtensions.Benchmarks;

/// <summary>
/// Compares strategies for extracting Mat pixel data into a managed byte array.
/// </summary>
[MemoryDiagnoser]
[Orderer(SummaryOrderPolicy.FastestToSlowest)]
public class MatToArrayBenchmarks
{
    /// <summary>
    /// Mat side lengths: 1080p, 4K.
    /// </summary>
    [Params(1920, 3840)]
    public int MatSize { get; set; }

    private Mat _mat = null!;

    [GlobalSetup]
    public void Setup()
    {
        _mat = new Mat(MatSize, MatSize, DepthType.Cv8U, 1);
        var span = _mat.GetSpan<byte>();
        for (int i = 0; i < span.Length; i++)
            span[i] = (byte)(i % 256);
    }

    [GlobalCleanup]
    public void Cleanup() => _mat.Dispose();

    /// <summary>
    /// Extension method: AllocateUninitializedArray + CopyTo.
    /// </summary>
    [Benchmark(Baseline = true, Description = "ToArray (extension)")]
    public byte[] ToArray() => _mat.ToArray();

    /// <summary>
    /// Emgu built-in raw data copy.
    /// </summary>
    [Benchmark(Description = "GetRawData")]
    public byte[] GetRawData() => _mat.GetRawData();

    /// <summary>
    /// Span-based copy via new byte[] allocation.
    /// </summary>
    [Benchmark(Description = "GetSpan.ToArray")]
    public byte[] SpanToArray() => _mat.GetSpan<byte>().ToArray();
}
