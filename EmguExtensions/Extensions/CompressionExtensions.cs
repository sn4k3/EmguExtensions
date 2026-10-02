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
    /// <remarks>
    /// This is an ascending scale (0 = <see cref="CompressionLevel.NoCompression"/>, 1 = <see cref="CompressionLevel.Fastest"/>,
    /// 2 = <see cref="CompressionLevel.Optimal"/>, 3+ = <see cref="CompressionLevel.SmallestSize"/>), it is NOT the numeric value of the
    /// <see cref="CompressionLevel"/> enum (where 0 is <see cref="CompressionLevel.Optimal"/> and 2 is <see cref="CompressionLevel.NoCompression"/>).
    /// </remarks>
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