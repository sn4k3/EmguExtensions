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

/// <summary>
/// Lossless TIFF compressor (OpenCV codec, Adobe Deflate). Unlike PNG it also stores signed, 32/64-bit integer and
/// 32/64-bit float depths.
/// </summary>
/// <remarks>Best for continuous-tone data (photos, depth maps, float fields): the TIFF horizontal predictor makes it
/// compress far better than the stream compressors (e.g. 1.45-1.7x versus 1.0-1.15x for Deflate/Brotli/Zstd). For
/// masks and images with large flat areas the stream compressors are smaller and much faster, prefer them there.
/// Native levels: 0 stores the image uncompressed inside the TIFF container, 1-9 are Deflate levels
/// (<see cref="CompressionLevel.Optimal"/> is 3: higher levels cost much more time for little gain).
/// Mats with an unsupported depth (<see cref="DepthType.Cv16F"/>, <see cref="DepthType.Cv16BF"/>,
/// <see cref="DepthType.CvBool"/>, FP8) or channel count (other than 1, 3 or 4) throw
/// <see cref="NotSupportedException"/>, which makes <see cref="CMat"/> store them raw. OpenCV writes
/// <see cref="DepthType.Cv64F"/> uncompressed regardless of the level, so it round trips but does not shrink.</remarks>
public sealed class MatCompressorTiff : MatCompressor
{
    /// <summary>
    /// Provides a singleton instance of the <see cref="MatCompressorTiff"/> class for efficient reuse across the application.
    /// </summary>
    public static readonly MatCompressorTiff Instance = new();

    /// <inheritdoc />
    private MatCompressorTiff() { }

    /// <inheritdoc />
    public override string Provider => "OpenCV";

    /// <inheritdoc />
    public override string Name => "TIFF";

    /// <inheritdoc />
    public override int MaximumCompressionLevel => 9;

    /// <inheritdoc />
    protected override bool AllocatesDestination => true;

    /// <inheritdoc />
    protected override int GetCompressionLevel(CompressionLevel compressionLevel)
    {
        return compressionLevel switch
        {
            CompressionLevel.NoCompression => 0,
            CompressionLevel.Fastest => 1,
            CompressionLevel.Optimal => 3,
            CompressionLevel.SmallestSize => 9,
            _ => throw new ArgumentOutOfRangeException(nameof(compressionLevel), compressionLevel, null),
        };
    }

    /// <inheritdoc />
    /// <exception cref="NotSupportedException">The depth or channel count cannot be stored losslessly as TIFF (the
    /// encoder silently converts or rejects them).</exception>
    protected override byte[] CompressCore(Mat src, int compressionLevel)
    {
        if (
            src.Depth
                is not (
                    DepthType.Cv8U
                    or DepthType.Cv8S
                    or DepthType.Cv16U
                    or DepthType.Cv16S
                    or DepthType.Cv32U
                    or DepthType.Cv32S
                    or DepthType.Cv64U
                    or DepthType.Cv64S
                    or DepthType.Cv32F
                    or DepthType.Cv64F
                )
            || src.NumberOfChannels is not (1 or 3 or 4)
        )
        {
            throw new NotSupportedException(
                $"TIFF cannot losslessly store {src.Depth} with {src.NumberOfChannels} channel(s)."
            );
        }

        return compressionLevel == 0
            ? src.GetTiffBytes(TiffCompression.None)
            : src.GetTiffBytes(TiffCompression.AdobeDeflate, compressionLevel);
    }

    /// <inheritdoc />
    protected override void DecompressCore(byte[] compressedBytes, Mat dst)
    {
        DecodeImage(compressedBytes, dst, Name);
    }
}
