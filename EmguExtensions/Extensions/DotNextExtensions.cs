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
using DotNext.Buffers;

namespace EmguExtensions;

/// <summary>
/// Provides extension methods for the DotNext library, enhancing the functionality of its types and enabling seamless integration.
/// </summary>
public static class DotNextExtensions
{
    /// <summary>
    /// Copies the contents of the <see cref="SparseBufferWriter{T}"/> to a new array and returns it.
    /// </summary>
    /// <param name="buffer">The buffer to copy from.</param>
    /// <typeparam name="T">The type of elements in the buffer.</typeparam>
    /// <returns>A new array containing the copied elements.</returns>
    public static T[] ToArray<T>(this SparseBufferWriter<T> buffer) where T : struct
    {
        var writtenCount = checked((int)buffer.WrittenCount);
        if (writtenCount == 0) return [];

        var output = GC.AllocateUninitializedArray<T>(writtenCount);
        buffer.CopyTo(output);
        return output;
    }
}