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
using Emgu.CV.Structure;

namespace EmguExtensions;

/// <summary>
/// Provides extension methods and utility functions for color manipulation and geometric calculations related to
/// regular polygons and their vertices.
/// </summary>
/// <remarks>This static class includes methods for scaling color channels, calculating polygon side lengths and
/// radii, and generating the vertices of regular polygons inscribed in ellipses or circles. The polygon vertex methods
/// support alignment, rotation, and mirroring options, and allow for precise control over rounding behavior when
/// converting to integer coordinates.</remarks>
public static class DrawingExtensions
{
    extension(Color color)
    {
        /// <summary>
        /// Creates a new <see cref="Color"/> by scaling each channel of this color by a pixel intensity value (0–255).
        /// </summary>
        /// <param name="pixelColor">The pixel intensity value (0–255) used as a scaling factor.</param>
        /// <param name="min">The minimum allowed value for each channel.</param>
        /// <param name="max">The maximum allowed value for each channel.</param>
        /// <returns>A new <see cref="Color"/> with each channel scaled by <paramref name="pixelColor"/> / 255.</returns>
        public Color FactorColor(byte pixelColor, byte min = 0, byte max = byte.MaxValue)
            => color.FactorColor(pixelColor / 255.0, min, max);

        /// <summary>
        /// Creates a new <see cref="Color"/> by multiplying each channel of this color by a factor, clamped between <paramref name="min"/> and <paramref name="max"/>. Channels that are zero remain zero.
        /// </summary>
        /// <param name="factor">The multiplicative factor to apply to each channel.</param>
        /// <param name="min">The minimum allowed value for each channel.</param>
        /// <param name="max">The maximum allowed value for each channel.</param>
        /// <returns>A new <see cref="Color"/> with each channel scaled by <paramref name="factor"/>.</returns>
        public Color FactorColor(double factor, byte min = 0, byte max = byte.MaxValue)
        {
            byte r = (byte)(color.R == 0 ? 0 :
                Math.Clamp(color.R * factor, min, max));

            byte g = (byte)(color.G == 0 ? 0 :
                Math.Clamp(color.G * factor, min, max));

            byte b = (byte)(color.B == 0 ? 0 :
                Math.Clamp(color.B * factor, min, max));
            return Color.FromArgb(color.A, r, g, b);
        }
    }


    /// <summary>
    /// Calculates the side length of a regular polygon given its circumscribed radius and number of sides.
    /// </summary>
    /// <param name="radius">The circumscribed radius of the polygon.</param>
    /// <param name="sides">The number of sides of the polygon.</param>
    /// <returns>The length of each side.</returns>
    public static double CalculatePolygonSideLengthFromRadius(double radius, int sides)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(sides, 1);
        return 2 * radius * Math.Sin(Math.PI / sides);
    }

    /// <summary>
    /// Calculates the apothem (vertical distance from center to the midpoint of a side) of a regular polygon.
    /// </summary>
    /// <param name="radius">The circumscribed radius of the polygon.</param>
    /// <param name="sides">The number of sides of the polygon.</param>
    /// <returns>The apothem length.</returns>
    public static double CalculatePolygonVerticalLengthFromRadius(double radius, int sides)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(sides, 1);
        return radius * Math.Cos(Math.PI / sides);
    }

    /// <summary>
    /// Calculates the circumscribed radius of a regular polygon given its side length and number of sides.
    /// </summary>
    /// <param name="length">The length of each side.</param>
    /// <param name="sides">The number of sides of the polygon.</param>
    /// <returns>The circumscribed radius.</returns>
    public static double CalculatePolygonRadiusFromSideLength(double length, int sides)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(sides, 1);
        var theta = 360.0 / sides;
        return length / (2 * Math.Cos((90 - theta / 2) * Math.PI / 180.0));
    }

    /// <summary>
    /// Computes the vertices of a regular polygon inscribed in an ellipse defined by <paramref name="diameter"/>.
    /// </summary>
    /// <param name="sides">The number of sides (must be at least 3).</param>
    /// <param name="diameter">The width and height of the bounding ellipse.</param>
    /// <param name="center">The center point of the polygon.</param>
    /// <param name="startingAngle">The starting rotation angle in degrees.</param>
    /// <param name="flipHorizontally">Whether to mirror the vertices horizontally.</param>
    /// <param name="flipVertically">Whether to mirror the vertices vertically.</param>
    /// <param name="midpointRounding">The rounding strategy used when converting to integer coordinates.</param>
    /// <returns>An array of <see cref="Point"/> representing the polygon vertices.</returns>
    public static Point[] GetPolygonVertices(int sides, SizeF diameter, PointF center, double startingAngle = 0, bool flipHorizontally = false, bool flipVertically = false, MidpointRounding midpointRounding = MidpointRounding.AwayFromZero)
    {
        if (sides < 3)
            throw new ArgumentException("Polygons can't have less than 3 sides...", nameof(sides));

        var vertices = new Point[sides];
        var radiusX = diameter.Width / 2; // X radius for pixel pitch
        var radiusY = diameter.Height / 2; // Y radius for pixel pitch

        if (sides == 4)
        {
            var rotatedRect = new RotatedRect(center, new SizeF(diameter.Width - 1, diameter.Height - 1), (float)startingAngle);
            var verticesF = rotatedRect.GetVertices();
            for (var i = 0; i < verticesF.Length; i++)
            {
                vertices[i] = verticesF[i].ToPoint(midpointRounding);
            }
        }
        else
        {
            var angleIncrement = 2 * Math.PI / sides;
            var startRotationAngleRadians = startingAngle * Math.PI / 180;

            for (int i = 0; i < sides; i++)
            {
                var angle = startRotationAngleRadians + i * angleIncrement;

                // Scale the X and Y coordinates independently for pixel pitch
                var x = (int)Math.Round(center.X + radiusX * Math.Cos(angle), midpointRounding);
                var y = (int)Math.Round(center.Y + radiusY * Math.Sin(angle), midpointRounding);

                vertices[i] = new Point(x, y);
            }
        }

        if (flipHorizontally)
        {
            var startX = center.X - radiusX;
            var endX = center.X + radiusX;
            for (int i = 0; i < sides; i++)
            {
                vertices[i].X = (int)Math.Round(endX - (vertices[i].X - startX), midpointRounding);
            }
        }

        if (flipVertically)
        {
            var startY = center.Y - radiusY;
            var endY = center.Y + radiusY;
            for (int i = 0; i < sides; i++)
            {
                vertices[i].Y = (int)Math.Round(endY - (vertices[i].Y - startY), midpointRounding);
            }
        }

        return vertices;
    }

    /// <summary>
    /// Computes the vertices of a regular polygon with a flat edge aligned to the bottom, inscribed in an ellipse defined by <paramref name="diameter"/>.
    /// </summary>
    /// <param name="sides">The number of sides (must be at least 3).</param>
    /// <param name="diameter">The width and height of the bounding ellipse.</param>
    /// <param name="center">The center point of the polygon.</param>
    /// <param name="startingAngle">An additional rotation angle in degrees applied after the alignment offset.</param>
    /// <param name="flipHorizontally">Whether to mirror the vertices horizontally.</param>
    /// <param name="flipVertically">Whether to mirror the vertices vertically.</param>
    /// <param name="midpointRounding">The rounding strategy used when converting to integer coordinates.</param>
    /// <returns>An array of <see cref="Point"/> representing the aligned polygon vertices.</returns>
    public static Point[] GetAlignedPolygonVertices(int sides, SizeF diameter, PointF center, double startingAngle = 0, bool flipHorizontally = false, bool flipVertically = false, MidpointRounding midpointRounding = MidpointRounding.AwayFromZero)
    {
        if (sides != 4) startingAngle += (180 - (360.0 / sides)) / 2;
        return GetPolygonVertices(sides, diameter, center, startingAngle, flipHorizontally, flipVertically, midpointRounding);
    }

    /// <summary>
    /// Computes the vertices of a regular polygon with a flat edge aligned to the bottom, inscribed in a circle of the given <paramref name="diameter"/>.
    /// </summary>
    /// <param name="sides">The number of sides (must be at least 3).</param>
    /// <param name="diameter">The diameter of the bounding circle.</param>
    /// <param name="center">The center point of the polygon.</param>
    /// <param name="startingAngle">An additional rotation angle in degrees applied after the alignment offset.</param>
    /// <param name="flipHorizontally">Whether to mirror the vertices horizontally.</param>
    /// <param name="flipVertically">Whether to mirror the vertices vertically.</param>
    /// <param name="midpointRounding">The rounding strategy used when converting to integer coordinates.</param>
    /// <returns>An array of <see cref="Point"/> representing the aligned polygon vertices.</returns>
    public static Point[] GetAlignedPolygonVertices(int sides, float diameter, PointF center, double startingAngle = 0, bool flipHorizontally = false, bool flipVertically = false, MidpointRounding midpointRounding = MidpointRounding.AwayFromZero)
    {
        if (sides != 4) startingAngle += (180 - (360.0 / sides)) / 2;
        return GetPolygonVertices(sides, new SizeF(diameter, diameter), center, startingAngle, flipHorizontally, flipVertically, midpointRounding);
    }
}
