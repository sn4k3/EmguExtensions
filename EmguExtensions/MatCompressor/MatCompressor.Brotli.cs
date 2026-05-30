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

using Emgu.CV;
using System.IO.Compression;

namespace EmguExtensions;

/// <inheritdoc />
public sealed class MatCompressorBrotli : MatCompressor
{
    // Constants fetched from System.IO.Compression.BrotliUtils.
    private const int WindowBitsMin = 10;
    private const int WindowBitsDefault = 22;
    private const int WindowBitsMax = 24;
    private const int QualityMin = 0;
    private const int QualityDefault = 4;
    private const int QualityMax = 11;

    /// <summary>
    /// Provides a singleton instance of the <see cref="MatCompressorBrotli"/> class for efficient reuse across the application.
    /// </summary>
    public static readonly MatCompressorBrotli Instance = new();

    /// <inheritdoc />
    public override string Name => "Brotli";

    /// <inheritdoc />
    public override int MaximumCompressionLevel => QualityMax;

    /// <inheritdoc />
    private MatCompressorBrotli() { }

    /// <inheritdoc />
    protected override int GetCompressionLevel(CompressionLevel compressionLevel)
    {
        return compressionLevel switch
        {
            CompressionLevel.NoCompression => QualityMin,
            CompressionLevel.Fastest => 1,
            CompressionLevel.Optimal => QualityDefault,
            CompressionLevel.SmallestSize => QualityMax,
            _ => throw new ArgumentException("Invalid CompressionLevel value.", nameof(compressionLevel))
        };
    }

    /// <inheritdoc />
    protected override byte[] CompressCore(Mat src, int compressionLevel)
    {
        var options = new BrotliCompressionOptions
        {
            Quality = compressionLevel
        };
        using var buffer = CreateCompressionBuffer(src);
        using (var brotliStream = new BrotliStream(CreateCompressionStream(buffer), options))
        {
            src.CopyTo(brotliStream);
        }

        return buffer.ToArray();
    }

    /// <inheritdoc />
    protected override void DecompressCore(byte[] compressedBytes, Mat dst)
    {
        var dstSpan = dst.GetSpan<byte>();
        if (!BrotliDecoder.TryDecompress(compressedBytes, dstSpan, out var bytesWritten))
        {
            throw new InvalidDataException("Failed to decompress Brotli data.");
        }

        if (bytesWritten != dstSpan.Length)
        {
            throw new InvalidDataException("Brotli decompressed size does not match destination Mat size.");
        }
    }
}
