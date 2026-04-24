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
public sealed class MatCompressorGZip : MatCompressor
{
    /// <summary>
    /// Provides a singleton instance of the <see cref="MatCompressorGZip"/> class for efficient reuse across the application.
    /// </summary>
    public static readonly MatCompressorGZip Instance = new();

    /// <inheritdoc />
    public override string Name => "GZip";

    /// <inheritdoc />
    private MatCompressorGZip() { }

    /// <inheritdoc />
    protected override byte[] CompressCore(Mat src, CompressionLevel compressionLevel)
    {
        using var compressedStream = EmguExtensions.RecyclableMemoryStreamManager.GetStream();
        using (var gzipStream = new GZipStream(compressedStream, compressionLevel, leaveOpen: true))
        {
            src.CopyTo(gzipStream);
        }

        return compressedStream.ToArrayPerf();
    }

    /// <inheritdoc />
    protected override void DecompressCore(byte[] compressedBytes, Mat dst)
    {
        unsafe
        {
            fixed (byte* pBuffer = compressedBytes)
            {
                using var compressedStream = new UnmanagedMemoryStream(pBuffer, compressedBytes.Length);
                using var gZipStream = new GZipStream(compressedStream, CompressionMode.Decompress, leaveOpen: true);
                gZipStream.ReadExactly(dst.GetSpan<byte>());
            }
        }
    }
}
