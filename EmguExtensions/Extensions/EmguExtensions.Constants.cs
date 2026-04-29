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

using Emgu.CV.Structure;
using System.Drawing;

namespace EmguExtensions;

/// <summary>
/// Provides a collection of constants and static properties for commonly used values in Emgu CV image processing operations, such as predefined colors and anchor points.
/// </summary>
public static partial class EmguExtensions
{
    /// <summary>
    /// Gets the scalar value representing the color black with full opacity.
    /// </summary>
    /// <remarks>This property provides a convenient way to reference the black color in image processing
    /// operations that use the MCvScalar structure. The value corresponds to a color with all channels set to zero
    /// except for the alpha channel, which is set to 255.</remarks>
    public static MCvScalar BlackColor => new(0, 0, 0, 255);

    /// <summary>
    /// Gets the scalar value representing pure white in the MCvScalar color space.
    /// </summary>
    /// <remarks>This property provides a convenient way to access a white color value with all channels set
    /// to their maximum (255). It can be used for drawing, masking, or initializing image regions where a white color
    /// is required.</remarks>
    public static MCvScalar WhiteColor => new(255, 255, 255, 255);

    /// <summary>
    /// Gets a Point structure representing the center anchor point, which is commonly used in various image processing
    /// </summary>
    public static Point AnchorCenter => new(-1, -1);

    /// <summary>
    /// Gets the scaling factor for normalizing byte pixel values to the range [0, 1]. This constant is commonly used when converting pixel values from byte format (0-255)
    /// to a normalized floating-point representation, which is often required for certain image processing algorithms and machine learning models.
    /// </summary>
    public const double NormalizedByteScale = 1.0 / 255.0;
}