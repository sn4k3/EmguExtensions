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
/// Provides extension methods for <see cref="MemoryStream"/>, including a method to create a copy of the stream's buffer without zeroing the memory for improved performance.
/// </summary>
public static class StreamExtensions
{
    extension(MemoryStream stream)
    {
        /// <summary>
        /// Returns the contents of the underlying stream as a byte array, using an optimized approach for
        /// performance.
        /// </summary>
        /// <remarks>This method may allocate an uninitialized array for performance when
        /// possible. The returned array is a copy of the stream's data and is not affected by subsequent changes to
        /// the stream.</remarks>
        /// <returns>A byte array containing the data from the stream. The array is empty if the stream has no data.</returns>
        public byte[] ToArrayPerf()
        {
            if (stream.Length == 0) return [];

            if (stream.TryGetBuffer(out var buffer))
            {
                var copy = GC.AllocateUninitializedArray<byte>(buffer.Count);
                buffer.AsSpan().CopyTo(copy);
                return copy;
            }

            return stream.ToArray();
        }
    }
}