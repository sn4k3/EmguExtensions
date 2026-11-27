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

namespace EmguExtensions;

/// <summary>
/// TIFF compression schemes (libtiff <c>COMPRESSION_*</c> values, OpenCV <c>ImwriteTiffCompressionFlags</c>) for
/// <c>GetTiffBytes</c>. Availability depends on the codecs built into the OpenCV runtime.
/// </summary>
public enum TiffCompression
{
    /// <summary>No compression.</summary>
    None = 1,

    /// <summary>Lempel-Ziv-Welch, lossless (OpenCV default for integer depths).</summary>
    Lzw = 5,

    /// <summary>JPEG, lossy.</summary>
    Jpeg = 7,

    /// <summary>Deflate (zlib) with the official Adobe tag, lossless. Supports levels 1-9.</summary>
    AdobeDeflate = 8,

    /// <summary>Macintosh run-length encoding, lossless.</summary>
    PackBits = 32773,

    /// <summary>Deflate (zlib) with the legacy tag, lossless. Supports levels 1-9.</summary>
    Deflate = 32946,

    /// <summary>LZMA2, lossless. Supports levels 0-9.</summary>
    Lzma = 34925,

    /// <summary>Zstandard, lossless. Supports levels 1-22.</summary>
    Zstd = 50000,

    /// <summary>WebP, lossy. The level is the quality, 1-100. Only 8-bit images with 1, 3 or 4 channels.</summary>
    Webp = 50001,
}
