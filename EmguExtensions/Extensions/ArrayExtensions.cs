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
/// Provides extension methods for arrays, including a method to create a copy of a byte array
/// without zeroing the memory for improved performance.
/// Use with caution as it may contain sensitive data.
/// </summary>
public static class ArrayExtensions
{
    extension(byte[] buffer)
    {
        /// <summary>
        /// Creates a copy of the byte array without zeroing the memory, which can be faster for large arrays. Use with caution as it may contain sensitive data.
        /// </summary>
        /// <returns></returns>
        public byte[] ToArrayPerf()
        {
            if (buffer.Length == 0) return [];
            var copy = GC.AllocateUninitializedArray<byte>(buffer.Length);
            buffer.CopyTo(copy);
            return copy;
        }
    }
}