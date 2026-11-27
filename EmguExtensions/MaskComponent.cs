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
/// A connected region of nonzero pixels in a mask, as returned by <c>GetMaskComponents</c>.
/// </summary>
/// <param name="Label">The connected-components label of the region (1-based, 0 is the background).</param>
/// <param name="Bounds">The bounding box of the region, in mask coordinates.</param>
/// <param name="Area">The number of pixels in the region.</param>
/// <param name="Centroid">The center of mass of the region, in mask coordinates.</param>
public readonly record struct MaskComponent(int Label, Rectangle Bounds, int Area, PointF Centroid);
