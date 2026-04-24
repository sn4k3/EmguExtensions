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
/// Provides extension methods for the Point and PointF structures, including distance calculation and rotation functionality.
/// </summary>
public static class PointExtensions
{
    /// <summary>
    /// Calculates the Euclidean distance between two points.
    /// </summary>
    /// <param name="start"></param>
    /// <param name="end"></param>
    /// <returns></returns>
    public static double FindLength(Point start, Point end)
    {
        double dx = end.X - start.X;
        double dy = end.Y - start.Y;
        return Math.Sqrt(dx * dx + dy * dy);
    }

    extension(Point point)
    {
        /// <summary>
        /// Rotates the Point by a specified angle in degrees around a given pivot point.
        /// </summary>
        /// <param name="angleDegree">The angle in degrees to rotate the point.</param>
        /// <param name="pivot">The pivot point around which to rotate. Defaults to the origin (0,0).</param>
        /// <returns>The rotated Point.</returns>
        public Point Rotate(double angleDegree, Point pivot = default)
        {
            if (angleDegree % 360 == 0) return point;
            double angle = angleDegree * Math.PI / 180;
            double cos = Math.Cos(angle);
            double sin = Math.Sin(angle);
            int dx = point.X - pivot.X;
            int dy = point.Y - pivot.Y;
            double x = cos * dx - sin * dy + pivot.X;
            double y = sin * dx + cos * dy + pivot.Y;

            return new((int)Math.Round(x), (int)Math.Round(y));
        }
    }

    extension(PointF point)
    {
        /// <summary>
        /// Rotates the PointF by a specified angle in degrees around a given pivot point.
        /// </summary>
        /// <param name="angleDegree">The angle in degrees to rotate the point.</param>
        /// <param name="pivot">The pivot point around which to rotate. Defaults to the origin (0,0).</param>
        /// <returns>The rotated PointF.</returns>
        public PointF Rotate(double angleDegree, PointF pivot = default)
        {
            if (angleDegree % 360 == 0) return point;
            double angle = angleDegree * Math.PI / 180;
            double cos = Math.Cos(angle);
            double sin = Math.Sin(angle);
            double dx = point.X - pivot.X;
            double dy = point.Y - pivot.Y;
            double x = cos * dx - sin * dy + pivot.X;
            double y = sin * dx + cos * dy + pivot.Y;

            return new((float)x, (float)y);
        }

        /// <summary>
        /// Converts the PointF to a Point by truncating the decimal part of the coordinates.
        /// </summary>
        /// <returns>A Point with coordinates truncated from the PointF.</returns>
        public Point ToPoint()
        {
            return new Point((int)point.X, (int)point.Y);
        }

        /// <summary>
        /// Converts the PointF to a Point using the specified MidpointRounding method for rounding the coordinates.
        /// </summary>
        /// <param name="rounding">The MidpointRounding method to use for rounding the coordinates.</param>
        /// <returns>A Point with coordinates rounded according to the specified MidpointRounding method.</returns>
        public Point ToPoint(MidpointRounding rounding)
        {
            return new Point((int)Math.Round(point.X, rounding), (int)Math.Round(point.Y, rounding));
        }
    }
}