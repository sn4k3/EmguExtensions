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

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System.Collections;
using System.Drawing;

namespace EmguExtensions;

/// <summary>
/// A contour cache for OpenCV
/// </summary>
public class EmguContour : LeaveOpenDisposableObject, IReadOnlyList<Point>, IComparable<EmguContour>, IComparer<EmguContour>
{
    #region Constants

    /// <summary>Hierarchy index for the next contour at the same level.</summary>
    public const byte HierarchyNextSameLevel = 0;

    /// <summary>Hierarchy index for the previous contour at the same level.</summary>
    public const byte HierarchyPreviousSameLevel = 1;

    /// <summary>Hierarchy index for the first child contour.</summary>
    public const byte HierarchyFirstChild = 2;

    /// <summary>Hierarchy index for the parent contour.</summary>
    public const byte HierarchyParent = 3;

    #endregion

    #region Members

    private readonly VectorOfPoint _vector;
    private Rectangle? _bounds;
    private RotatedRect? _boundsBestFit;
    private CircleF? _minEnclosingCircle;
    private bool? _isConvex;
    private Moments? _moments;
    private Point? _centroid;

    #endregion

    #region Properties

    /// <summary>
    /// Gets the minimum X coordinate of the bounding rectangle.
    /// </summary>
    public int XMin => BoundingRectangle.X;

    /// <summary>
    /// Gets the minimum Y coordinate of the bounding rectangle.
    /// </summary>
    public int YMin => BoundingRectangle.Y;

    /// <summary>
    /// Gets the maximum X coordinate (exclusive) of the bounding rectangle.
    /// </summary>
    public int XMax => BoundingRectangle.Right;

    /// <summary>
    /// Gets the maximum Y coordinate (exclusive) of the bounding rectangle.
    /// </summary>
    public int YMax => BoundingRectangle.Bottom;

    /// <summary>
    /// Gets the axis-aligned bounding rectangle of the contour.
    /// </summary>
    public Rectangle BoundingRectangle
    {
        get
        {
            ThrowIfDisposed();
            return _bounds ??= CvInvoke.BoundingRectangle(_vector);
        }
    }

    /// <summary>
    /// Gets the minimum-area rotated bounding rectangle of the contour.
    /// </summary>
    public RotatedRect BoundsBestFit
    {
        get
        {
            ThrowIfDisposed();
            return _boundsBestFit ??= CvInvoke.MinAreaRect(_vector);
        }
    }

    /// <summary>
    /// Gets the minimum enclosing circle of the contour.
    /// </summary>
    public CircleF MinEnclosingCircle
    {
        get
        {
            ThrowIfDisposed();
            return _minEnclosingCircle ??= CvInvoke.MinEnclosingCircle(_vector);
        }
    }

    /// <summary>
    /// Gets whether the contour is convex.
    /// </summary>
    public bool IsConvex
    {
        get
        {
            ThrowIfDisposed();
            return _isConvex ??= CvInvoke.IsContourConvex(_vector);
        }
    }

    /// <summary>
    /// Gets the area of the contour.
    /// </summary>
    public double Area
    {
        get
        {
            ThrowIfDisposed();
            if (double.IsNaN(field))
            {
                field = CvInvoke.ContourArea(_vector);
            }

            return field;
        }
    } = double.NaN;

    /// <summary>
    /// Gets the perimeter of the contour.
    /// </summary>
    public double Perimeter
    {
        get
        {
            ThrowIfDisposed();
            if (double.IsNaN(field))
            {
                field = CvInvoke.ArcLength(_vector, true);
            }
            return field;
        }
    } = double.NaN;

    /// <summary>
    /// Gets whether the contour is closed.
    /// </summary>
    public bool IsClosed => IsConvex || Area > Perimeter;

    /// <summary>
    /// Gets whether the contour is open.
    /// </summary>
    public bool IsOpen => !IsClosed;

    /// <summary>
    /// Gets the spatial moments of the contour.
    /// </summary>
    public Moments Moments
    {
        get
        {
            ThrowIfDisposed();
            return _moments ??= CvInvoke.Moments(_vector);
        }
    }

    /// <summary>
    /// Gets the centroid of the contour, or <see cref="EmguExtensions.AnchorCenter"/> if the contour has zero area.
    /// </summary>
    public Point Centroid
    {
        get
        {
            ThrowIfDisposed();
            return _centroid ??= Moments.M00 == 0 ? EmguExtensions.AnchorCenter :
                new Point(
                    (int)Math.Round(Moments.M10 / Moments.M00),
                    (int)Math.Round(Moments.M01 / Moments.M00));
        }
    }

    /// <summary>
    /// Gets the contour points as a <see cref="VectorOfPoint"/>.
    /// </summary>
    public VectorOfPoint Vector
    {
        get
        {
            ThrowIfDisposed();
            return _vector;
        }
    }

    /// <summary>
    /// Gets whether this contour has no points.
    /// </summary>
    public bool IsEmpty => Vector.Size == 0;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of the <see cref="EmguContour"/> class with the specified contour points.
    /// </summary>
    /// <param name="points">The contour points.</param>
    /// <param name="leaveOpen">Indicates whether to dispose the vector when the contour is disposed.</param>
    public EmguContour(VectorOfPoint points, bool leaveOpen = true) : base(leaveOpen)
    {
        _vector = points;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="EmguContour"/> class with the specified contour points.
    /// </summary>
    /// <param name="points">The contour points.</param>
    public EmguContour(Point[] points) : this(new VectorOfPoint(points), false)
    {
    }
    #endregion

    #region Methods

    /// <summary>
    /// Checks if a given <see cref="Point"/> is inside the contour rectangle bounds
    /// </summary>
    /// <param name="point">The point to check.</param>
    /// <returns>True if the point is inside the contour bounds; otherwise, false.</returns>
    public bool IsInsideBounds(Point point) => BoundingRectangle.Contains(point);

    /// <summary>
    /// Checks whether a given <see cref="Point"/> is inside the contour using a point-in-polygon test.
    /// </summary>
    /// <param name="point">The point to check.</param>
    /// <returns><see langword="true"/> if the point is inside or on the contour edge; otherwise <see langword="false"/>.</returns>
    public bool IsInside(Point point)
    {
        if (!IsInsideBounds(point)) return false;
        return CvInvoke.PointPolygonTest(Vector, point, false) >= 0;
    }

    /// <summary>
    /// Measures the signed distance from a point to the nearest contour edge.
    /// </summary>
    /// <param name="point">The point to measure from.</param>
    /// <returns>Positive if inside, negative if outside, zero if exactly on the edge.</returns>
    public double MeasureDist(Point point)
    {
        return CvInvoke.PointPolygonTest(Vector, point, true);
    }

    /// <summary>
    /// Approximates the contour shape with fewer vertices using the Douglas-Peucker algorithm.
    /// </summary>
    /// <param name="epsilon">Approximation accuracy as a fraction of the contour perimeter.</param>
    /// <returns>A new <see cref="Mat"/> containing the approximated contour. The caller is responsible for disposing it.</returns>
    public Mat ContourApproximation(double epsilon = 0.1)
    {
        var mat = new Mat();
        CvInvoke.ApproxPolyDP(Vector, mat, epsilon * Perimeter, true);
        return mat;
    }

    /// <summary>
    /// Draws the minimum enclosing circle of this contour onto the specified image.
    /// </summary>
    /// <param name="src">The image to draw on.</param>
    /// <param name="color">The circle color.</param>
    /// <param name="thickness">The circle line thickness. Use -1 for filled.</param>
    /// <param name="lineType">The line drawing type.</param>
    /// <param name="shift">Number of fractional bits in the point coordinates.</param>
    public void FitCircle(Mat src, MCvScalar color, int thickness = 1, LineType lineType = LineType.EightConnected, int shift = 0)
    {
        CvInvoke.Circle(src,
            new Point((int)MinEnclosingCircle.Center.X, (int)MinEnclosingCircle.Center.Y),
            (int) Math.Round(MinEnclosingCircle.Radius),
            color,
            thickness,
            lineType,
            shift);
    }

    #endregion

    #region Static methods

    /// <summary>
    /// Computes the centroid of the specified contour points.
    /// </summary>
    /// <param name="points">The contour points.</param>
    /// <returns>The centroid point, or <see cref="EmguExtensions.AnchorCenter"/> if the contour is empty or has zero area.</returns>
    public static Point GetCentroid(VectorOfPoint points)
    {
        if (points.Length == 0) return EmguExtensions.AnchorCenter;
        using var moments = CvInvoke.Moments(points);
        return moments.M00 == 0 ? EmguExtensions.AnchorCenter :
            new Point(
                (int)Math.Round(moments.M10 / moments.M00),
                (int)Math.Round(moments.M01 / moments.M00));
    }
    #endregion

    #region Implementations

    /// <summary>
    /// Creates a deep copy of this contour with independent point data.
    /// </summary>
    /// <returns>A new <see cref="EmguContour"/> instance.</returns>
    public EmguContour Clone()
    {
        return new EmguContour(Vector.ToArray());
    }

    /// <inheritdoc />
    public IEnumerator<Point> GetEnumerator()
    {
        ThrowIfDisposed();
        return Enumerate(_vector).GetEnumerator();

        static IEnumerable<Point> Enumerate(VectorOfPoint vector)
        {
            for (int i = 0; i < vector.Size; i++)
            {
                yield return vector[i];
            }
        }
    }

    /// <inheritdoc />
    IEnumerator IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }

    /// <inheritdoc />
    public int Count => Vector.Size;

    /// <inheritdoc />
    public Point this[sbyte index] => Vector[index];
    /// <inheritdoc />
    public Point this[byte index] => Vector[index];
    /// <inheritdoc />
    public Point this[short index] => Vector[index];
    /// <inheritdoc />
    public Point this[ushort index] => Vector[index];
    /// <inheritdoc />
    public Point this[int index] => Vector[index];
    /// <inheritdoc />
    public Point this[uint index] => Vector[(int) index];
    /// <inheritdoc />
    public Point this[long index] => Vector[(int) index];
    /// <inheritdoc />
    public Point this[ulong index] => Vector[(int) index];

    /// <inheritdoc />
    protected override void DisposeManaged()
    {
        if (!LeaveOpen) _vector.Dispose();
        _moments?.Dispose();
        _moments = null;
    }

    #endregion

    #region Equality

    /// <inheritdoc />
    public override int GetHashCode()
    {
        if (Count == 0) return 0;
        int lastIndex = Count - 1;
        return HashCode.Combine(Count, Vector[0], Vector[lastIndex / 2], Vector[lastIndex]);
    }

    /// <summary>
    /// Determines whether this contour has the same points as another contour.
    /// </summary>
    /// <param name="other">The contour to compare with.</param>
    /// <returns><see langword="true"/> if all points match; otherwise <see langword="false"/>.</returns>
    protected bool Equals(EmguContour other)
    {
        if (Count != other.Count) return false;
        for (var i = 0; i < Count; i++)
        {
            if (Vector[i] != other.Vector[i]) return false;
        }

        return true;
    }

    /// <inheritdoc />
    public override bool Equals(object? obj)
    {
        if (ReferenceEquals(null, obj)) return false;
        if (ReferenceEquals(this, obj)) return true;
        if (obj.GetType() != this.GetType()) return false;
        return Equals((EmguContour) obj);
    }

    /// <inheritdoc />
    public int CompareTo(EmguContour? other)
    {
        if (ReferenceEquals(this, other)) return 0;
        if (ReferenceEquals(null, other)) return 1;
        return Area.CompareTo(other.Area);
    }

    /// <inheritdoc />
    public int Compare(EmguContour? x, EmguContour? y)
    {
        if (ReferenceEquals(x, y)) return 0;
        if (ReferenceEquals(null, y)) return 1;
        if (ReferenceEquals(null, x)) return -1;
        return x.Area.CompareTo(y.Area);
    }

    #endregion
}
