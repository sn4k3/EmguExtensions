/*
 *   MIT License
 *
 *   Copyright (c) 2026 Tiago Conceição
 *
 *   Permission is hereby granted, free of charge, to any person obtaining a copy
 *   of this software and associated documentation files (the "Software"), to deal
 *   in the Software without restriction, including without limitation the rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:
 *
 *   The above copyright notice and this permission notice shall be included in all
 *   copies or substantial portions of the Software.
 *
 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *   SOFTWARE.
 */

#if NET11_0_OR_GREATER
using Emgu.CV;
using System.IO.Compression;

namespace EmguExtensions;

/// <inheritdoc />
public sealed class MatCompressorZstd : MatCompressor
{
    // Quality limits and defaults are provided by the native Zstandard library.
    private static readonly int QualityMin = ZstandardCompressionOptions.MinQuality;
    private static readonly int QualityDefault = ZstandardCompressionOptions.DefaultQuality;
    private static readonly int QualityMax = ZstandardCompressionOptions.MaxQuality;
    private const int QualityFastest = 1;

    /// <summary>
    /// Provides a singleton instance of the <see cref="MatCompressorZstd"/> class for efficient reuse across the application.
    /// </summary>
    public static readonly MatCompressorZstd Instance = new();

    /// <inheritdoc />
    private MatCompressorZstd() { }

    /// <inheritdoc />
    public override string Name => "Zstandard";

    /// <inheritdoc />
    public override int MinimumCompressionLevel => QualityMin;

    /// <inheritdoc />
    public override int MaximumCompressionLevel => QualityMax;

    /// <inheritdoc />
    protected override int GetCompressionLevel(CompressionLevel compressionLevel)
    {
        return compressionLevel switch
        {
            CompressionLevel.NoCompression => QualityMin,
            CompressionLevel.Fastest => QualityFastest,
            CompressionLevel.Optimal => QualityDefault,
            CompressionLevel.SmallestSize => QualityMax,
            _ => throw new ArgumentOutOfRangeException(
                nameof(compressionLevel),
                compressionLevel,
                null
            ),
        };
    }

    /// <inheritdoc />
    protected override byte[] CompressCore(Mat src, int compressionLevel)
    {
        var options = new ZstandardCompressionOptions { Quality = compressionLevel };
        using var buffer = CreateCompressionBuffer(src);
        using (var zstdStream = new ZstandardStream(CreateCompressionStream(buffer), options))
        {
            src.CopyTo(zstdStream);
        }

        return buffer.ToArray();
    }

    /// <inheritdoc />
    protected override void DecompressCore(byte[] compressedBytes, Mat dst)
    {
        using var compressedStream = new MemoryStream(compressedBytes, writable: false);
        using var zstdStream = new ZstandardStream(
            compressedStream,
            CompressionMode.Decompress,
            leaveOpen: true
        );

        if (dst.IsContinuous)
        {
            zstdStream.ReadExactly(dst.GetSpan<byte>());
        }
        else
        {
            for (var row = 0; row < dst.Height; row++)
            {
                zstdStream.ReadExactly(dst.GetRowSpanOfBytes(row));
            }
        }
    }
}
#endif
