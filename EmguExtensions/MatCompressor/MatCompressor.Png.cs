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

using System.IO.Compression;
using Emgu.CV;
using Emgu.CV.CvEnum;

namespace EmguExtensions;

/// <inheritdoc />
public sealed class MatCompressorPng : MatCompressor
{
    /// <summary>
    /// Provides a singleton instance of the <see cref="MatCompressorPng"/> class for efficient reuse across the application.
    /// </summary>
    public static readonly MatCompressorPng Instance = new();

    /// <inheritdoc />
    private MatCompressorPng() { }

    /// <inheritdoc />
    public override string Provider => "OpenCV";

    /// <inheritdoc />
    public override string Name => "PNG";

    /// <inheritdoc />
    public override int MaximumCompressionLevel => 9;

    /// <inheritdoc />
    protected override int GetCompressionLevel(CompressionLevel compressionLevel)
    {
        return compressionLevel switch
        {
            CompressionLevel.NoCompression => 0,
            CompressionLevel.Fastest => 1,
            CompressionLevel.Optimal => 3,
            CompressionLevel.SmallestSize => 9,
            _ => throw new ArgumentOutOfRangeException(nameof(compressionLevel), compressionLevel, null)
        };
    }

    /// <inheritdoc />
    protected override byte[] CompressCore(Mat src, int compressionLevel)
    {
        return src.GetPngBytes(compressionLevel);
    }

    /// <inheritdoc />
    protected override void DecompressCore(byte[] compressedBytes, Mat dst)
    {
        CvInvoke.Imdecode(compressedBytes, ImreadModes.Unchanged, dst);
    }
}
