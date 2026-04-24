using System.IO.Compression;
using BenchmarkDotNet.Filters;
using BenchmarkDotNet.Running;

namespace EmguExtensions.Benchmarks;

/// <summary>
/// Filters out benchmark cases where the <see cref="CompressionLevel"/> has no effect on the compressor.
/// For example, <c>None</c> produces identical results at every level, so only <see cref="CompressionLevel.NoCompression"/> is kept.
/// </summary>
public class RedundantCompressorLevelFilter : IFilter
{
    /// <summary>
    /// Compressor names for which only <see cref="CompressionLevel.NoCompression"/> is meaningful.
    /// </summary>
    private static readonly HashSet<string> LevelAgnosticCompressors = [nameof(MatCompressorNone)];

    static RedundantCompressorLevelFilter()
    {
        // Populate from the registered compressors: any whose name matches a known level-agnostic type.
        // Currently only "None", but this is extensible.
        foreach (var compressor in MatCompressor.AvailableCompressors)
        {
            if (compressor is MatCompressorNone)
                LevelAgnosticCompressors.Add(compressor.Name);
        }
    }

    public bool Predicate(BenchmarkCase benchmarkCase)
    {
        string? compressorName = null;
        CompressionLevel? level = null;

        foreach (var param in benchmarkCase.Parameters.Items)
        {
            if (param.Name == nameof(MatCompressorBenchmarks.CompressorName))
                compressorName = param.Value as string;
            else if (param.Name == nameof(MatCompressorBenchmarks.Level))
                level = param.Value as CompressionLevel?;
        }

        if (compressorName is not null
            && level is not null
            && LevelAgnosticCompressors.Contains(compressorName)
            && level != CompressionLevel.NoCompression)
        {
            return false; // Skip this combination
        }

        return true;
    }
}
