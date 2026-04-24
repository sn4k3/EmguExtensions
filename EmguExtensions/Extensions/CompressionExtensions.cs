using System.IO.Compression;

namespace EmguExtensions;

/// <summary>
/// Provides extension methods for mapping integer compression levels to
/// the <see cref="CompressionLevel"/> enum values.
/// </summary>
public static class CompressionExtensions
{
    /// <summary>
    /// Maps an integer compression level (0-3) to the corresponding <see cref="CompressionLevel"/> enum value.
    /// </summary>
    /// <param name="level">The integer compression level (0-3).</param>
    /// <returns>The corresponding <see cref="CompressionLevel"/> enum value.</returns>
    public static CompressionLevel GetCompressionLevel(int level)
    {
        return level switch
        {
            0 => CompressionLevel.NoCompression,
            1 => CompressionLevel.Fastest,
            2 => CompressionLevel.Optimal,
            _ when level >= 3 => CompressionLevel.SmallestSize,
            _ => throw new ArgumentOutOfRangeException(nameof(level), level, "Compression level must be non-negative.")
        };
    }
}