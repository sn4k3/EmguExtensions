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

using System.Drawing;

namespace EmguExtensions;

/// <summary>
/// Represents a contiguous run of pixels with the same grey value produced by <see cref="EmguExtensions"/> stride scans.
/// </summary>
/// <param name="Index">Zero-based offset of the first pixel of the stride in the source span.</param>
/// <param name="Location">Coordinates of the first pixel of the stride in the source image.</param>
/// <param name="Stride">Number of consecutive pixels sharing <paramref name="Grey"/>.</param>
/// <param name="Grey">Grey value shared by every pixel in the stride.</param>
public readonly record struct GreyStride(int Index, Point Location, uint Stride, byte Grey);
