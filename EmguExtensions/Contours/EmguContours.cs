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

using System.Collections;
using System.Collections.ObjectModel;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Util;

namespace EmguExtensions;

/// <summary>
/// A disposable contour collection that wraps an OpenCV contour hierarchy, providing indexed access
/// to individual <see cref="EmguContour"/> instances and a tree of <see cref="EmguContourFamily"/> nodes.
/// </summary>
/// <remarks>Best used with <see cref="RetrType.Tree"/> contour retrieval mode for full hierarchy support.</remarks>
public class EmguContours : LeaveOpenDisposableObject, IReadOnlyList<EmguContour>
{
    #region Members

    /// <summary>
    /// The array of <see cref="EmguContour"/> instances corresponding to each contour vector
    /// in the hierarchy.
    /// </summary>
    private readonly EmguContour[] _contours;

    #endregion

    #region Properties

    /// <summary>
    /// Gets the underlying <see cref="VectorOfVectorOfPoint"/> holding all contour point data.
    /// </summary>
    public VectorOfVectorOfPoint Vector { get; }

    /// <summary>
    /// Gets the contour hierarchy matrix where each row contains [next, previous, first_child, parent] indices.
    /// </summary>
    public readonly int[,] Hierarchy;

    /// <summary>
    /// Gets the count of external (root-level) contours.
    /// </summary>
    public int ExternalContoursCount => Families.Count;

    /// <summary>
    /// Gets the read-only collection of top-level contour families.
    /// </summary>
    public ReadOnlyCollection<EmguContourFamily> Families { get; }

    /// <summary>
    /// Gets whether this collection has no contours.
    /// </summary>
    public bool IsEmpty => Count == 0;

    /// <summary>
    /// Gets the external (root-level) contours.
    /// </summary>
    public IEnumerable<EmguContour> ExternalContours => Families.Select(family => family.Self);

    /// <summary>
    /// Gets the total solid area enclosed in all contour families,
    /// which is the sum of all positive areas minus the sum of all negative areas.
    /// </summary>
    /// <remarks>Lazily computed and cached on first access. Only works with <see cref="RetrType.Tree"/> contour detection mode.</remarks>
    public double TotalSolidArea
    {
        get
        {
            if (IsEmpty) return 0;
            if (double.IsNaN(field)) field = Families.Sum(family => family.TotalSolidArea);
            return field;
        }
    } = double.NaN;

    /// <summary>
    /// Gets the minimum <see cref="EmguContourFamily.TotalSolidArea"/> among all top-level contour families.
    /// </summary>
    /// <remarks>Lazily computed and cached on first access. Only works with <see cref="RetrType.Tree"/> contour detection mode.</remarks>
    public double MinSolidArea
    {
        get
        {
            if (IsEmpty) return 0;
            if (double.IsNaN(field)) field = Families.Min(family => family.TotalSolidArea);
            return field;
        }
    } = double.NaN;

    /// <summary>
    /// Gets the maximum <see cref="EmguContourFamily.TotalSolidArea"/> among all top-level contour families.
    /// </summary>
    /// <remarks>Lazily computed and cached on first access. Only works with <see cref="RetrType.Tree"/> contour detection mode.</remarks>
    public double MaxSolidArea
    {
        get
        {
            if (IsEmpty) return 0;
            if (double.IsNaN(field)) field = Families.Max(family => family.TotalSolidArea);
            return field;
        }
    } = double.NaN;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of the <see cref="EmguContours"/> class from raw contour point arrays and hierarchy data.
    /// </summary>
    /// <param name="points">The contour points as jagged arrays.</param>
    /// <param name="hierarchy">The contour hierarchy matrix.</param>
    public EmguContours(Point[][] points, int[,] hierarchy) : this(new VectorOfVectorOfPoint(points), hierarchy)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="EmguContours"/> class from pre-built contour vectors and hierarchy data.
    /// </summary>
    /// <param name="vectorOfPointsOfPoints">The contour vectors.</param>
    /// <param name="hierarchy">The contour hierarchy matrix.</param>
    /// <param name="leaveOpen">If <see langword="true"/>, the <paramref name="vectorOfPointsOfPoints"/> will not be disposed when this instance is disposed.</param>
    public EmguContours(VectorOfVectorOfPoint vectorOfPointsOfPoints, int[,] hierarchy, bool leaveOpen = false) : base(leaveOpen)
    {
        Vector = vectorOfPointsOfPoints;
        Hierarchy = hierarchy;
        _contours = new EmguContour[Vector.Size];
        Families = BuildFamilies();
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="EmguContours"/> class by finding contours in the specified image.
    /// </summary>
    /// <param name="mat">The input image to find contours in.</param>
    /// <param name="mode">The contour retrieval mode.</param>
    /// <param name="method">The contour approximation method.</param>
    /// <param name="offset">Optional offset applied to every contour point.</param>
    public EmguContours(IInputOutputArray mat, RetrType mode = RetrType.Tree, ChainApproxMethod method = ChainApproxMethod.ChainApproxSimple, Point offset = default)
    {
        Vector = mat.FindContours(out Hierarchy, mode, method, offset);
        _contours = new EmguContour[Vector.Size];
        Families = BuildFamilies();
    }

    /// <summary>
    /// Builds the family tree from the contour hierarchy.
    /// </summary>
    /// <returns>A read-only collection of top-level contour families.</returns>
    private ReadOnlyCollection<EmguContourFamily> BuildFamilies()
    {
        List<EmguContourFamily> families = [];

        for (int i = 0; i < Count; i++)
        {
            if (Hierarchy[i, EmguContour.HierarchyParent] == -1)
            {
                families.Add(ArrangeFamily(i, 0, null));
            }
        }

        return families.AsReadOnly();
    }

    /// <summary>
    /// Recursively arranges a contour and its children into a <see cref="EmguContourFamily"/> tree node.
    /// </summary>
    /// <param name="contourIndex">The contour index to arrange.</param>
    /// <param name="depth">The depth level in the hierarchy (0 = external/root).</param>
    /// <param name="parent">The parent family node, or <see langword="null"/> for root contours.</param>
    /// <returns>The constructed family node with all children attached.</returns>
    private EmguContourFamily ArrangeFamily(int contourIndex, int depth, EmguContourFamily? parent)
    {
        _contours[contourIndex] = new EmguContour(Vector[contourIndex]);
        var family = new EmguContourFamily(contourIndex, depth, _contours[contourIndex], parent);

        for (int childIndex = Hierarchy[contourIndex, EmguContour.HierarchyFirstChild];
             childIndex >= 0;
             childIndex = Hierarchy[childIndex, EmguContour.HierarchyNextSameLevel])
        {
            family.AddChild(ArrangeFamily(childIndex, depth + 1, family));
        }

        return family;
    }

    /// <summary>
    /// Gets the contours that contain the specified location.
    /// </summary>
    /// <param name="location">The point to test against.</param>
    /// <param name="includeLimitingArea">If <see langword="true"/>, includes all nested child contours; otherwise only the outermost containing contour is returned.</param>
    /// <returns>A new <see cref="VectorOfVectorOfPoint"/> with the matching contours. The caller is responsible for disposing it.</returns>
    public VectorOfVectorOfPoint GetContoursInside(Point location, bool includeLimitingArea = true)
    {
        return GetContoursInside(Vector, Hierarchy, location, includeLimitingArea);
    }
    #endregion

    #region Indexers

    /// <summary>Gets the contour at the specified index.</summary>
    public EmguContour this[sbyte index] => _contours[index];
    /// <summary>Gets the contour at the specified index.</summary>
    public EmguContour this[byte index] => _contours[index];
    /// <summary>Gets the contour at the specified index.</summary>
    public EmguContour this[short index] => _contours[index];
    /// <summary>Gets the contour at the specified index.</summary>
    public EmguContour this[ushort index] => _contours[index];
    /// <inheritdoc />
    public EmguContour this[int index] => _contours[index];
    /// <summary>Gets the contour at the specified index.</summary>
    public EmguContour this[uint index] => _contours[index];
    /// <summary>Gets the contour at the specified index.</summary>
    public EmguContour this[ulong index] => _contours[index];
    /// <summary>Gets the contour at the specified index.</summary>
    public EmguContour this[long index] => _contours[index];

    /// <summary>Gets a hierarchy value for the contour at the specified index.</summary>
    public int this[sbyte index, int hierarchyIndex] => Hierarchy[index, hierarchyIndex];
    /// <summary>Gets a hierarchy value for the contour at the specified index.</summary>
    public int this[byte index, int hierarchyIndex] => Hierarchy[index, hierarchyIndex];
    /// <summary>Gets a hierarchy value for the contour at the specified index.</summary>
    public int this[short index, int hierarchyIndex] => Hierarchy[index, hierarchyIndex];
    /// <summary>Gets a hierarchy value for the contour at the specified index.</summary>
    public int this[ushort index, int hierarchyIndex] => Hierarchy[index, hierarchyIndex];
    /// <summary>Gets a hierarchy value for the contour at the specified index.</summary>
    public int this[int index, int hierarchyIndex] => Hierarchy[index, hierarchyIndex];
    /// <summary>Gets a hierarchy value for the contour at the specified index.</summary>
    public int this[uint index, int hierarchyIndex] => Hierarchy[index, hierarchyIndex];
    /// <summary>Gets a hierarchy value for the contour at the specified index.</summary>
    public int this[ulong index, int hierarchyIndex] => Hierarchy[index, hierarchyIndex];
    /// <summary>Gets a hierarchy value for the contour at the specified index.</summary>
    public int this[long index, int hierarchyIndex] => Hierarchy[index, hierarchyIndex];

    #endregion

    #region IReadOnlyList Implementation

    /// <inheritdoc />
    public IEnumerator<EmguContour> GetEnumerator()
    {
        return ((IEnumerable<EmguContour>)_contours).GetEnumerator();
    }

    /// <inheritdoc />
    IEnumerator IEnumerable.GetEnumerator()
    {
        return _contours.GetEnumerator();
    }

    /// <inheritdoc />
    public int Count => _contours.Length;

    #endregion

    #region Methods

    /// <summary>
    /// Calculates the Euclidean distance between the centroid of each contour and every other contour.
    /// </summary>
    /// <param name="includeOwn">If <see langword="true"/>, each contour's own entry (distance 0) is included in its result array.</param>
    /// <param name="sortByDistance">If <see langword="true"/>, each contour's result array is sorted by ascending distance.</param>
    /// <returns>A jagged array indexed by contour, where each inner array contains tuples of (index, contour, distance) pairs.</returns>
    public (int Index, EmguContour Contour, double Distance)[][] CalculateCentroidDistances(bool includeOwn = false, bool sortByDistance = true)
    {
        var items = new (int Index, EmguContour Contour, double Distance)[Count][];
        for (int i = 0; i < Count; i++)
        {
            items[i] = new (int Index, EmguContour Contour, double Distance)[includeOwn ? Count : Count - 1];
            int count = 0;
            for (int x = 0; x < Count; x++)
            {
                if (x == i && !includeOwn) continue;

                items[i][count] = new(x, this[x],
                    x == i ? 0 : PointExtensions.FindLength(this[i].Centroid, this[x].Centroid));
                count++;
            }

            if (sortByDistance) Array.Sort(items[i], (left, right) => left.Distance.CompareTo(right.Distance));
        }

        return items;
    }

    /// <summary>
    /// Creates a deep copy of this <see cref="EmguContours"/> instance with independent contour data.
    /// </summary>
    /// <returns>A new <see cref="EmguContours"/> instance.</returns>
    public EmguContours Clone()
    {
        return new EmguContours(Vector.ToArrayOfArray(), (int[,])Hierarchy.Clone());
    }

    /// <inheritdoc />
    protected override void DisposeManaged()
    {
        if (!LeaveOpen)
        {
            Vector.Dispose();
        }

        foreach (var contour in _contours)
        {
            contour.Dispose();
        }
    }

    #endregion

    #region Static methods

    /// <summary>
    /// Gets the contours that contain the specified location.
    /// </summary>
    /// <param name="contours">The contour vectors to search.</param>
    /// <param name="hierarchy">The contour hierarchy matrix.</param>
    /// <param name="location">The point to test against.</param>
    /// <param name="includeLimitingArea">If <see langword="true"/>, includes all nested child contours; otherwise only the outermost containing contour is returned.</param>
    /// <returns>A new <see cref="VectorOfVectorOfPoint"/> with the matching contours. The caller is responsible for disposing it.</returns>
    public static VectorOfVectorOfPoint GetContoursInside(VectorOfVectorOfPoint contours, int[,] hierarchy, Point location, bool includeLimitingArea = true)
    {
        var vector = new VectorOfVectorOfPoint();
        var vectorSize = contours.Size;
        for (var i = vectorSize - 1; i >= 0; i--)
        {
            if (CvInvoke.PointPolygonTest(contours[i], location, false) < 0) continue;
            vector.Push(contours[i]);
            if (!includeLimitingArea) break;
            for (int n = i + 1; n < vectorSize; n++)
            {
                if (hierarchy[n, EmguContour.HierarchyParent] != i) continue;
                vector.Push(contours[n]);
            }
            break;
        }

        return vector;
    }

    /// <summary>
    /// Gets the innermost contour that contains the specified location.
    /// </summary>
    /// <param name="contours">The contour vectors to search.</param>
    /// <param name="location">The point to test against.</param>
    /// <param name="index">The index of the found contour, or -1 if no contour contains the location.</param>
    /// <returns>The matching <see cref="VectorOfPoint"/>, or <see langword="null"/> if no contour contains the location.</returns>
    public static VectorOfPoint? GetContourInside(VectorOfVectorOfPoint contours, Point location, out int index)
    {
        index = -1;
        var vectorSize = contours.Size;
        for (int i = vectorSize - 1; i >= 0; i--)
        {
            if (CvInvoke.PointPolygonTest(contours[i], location, false) < 0) continue;
            index = i;
            return contours[i];
        }

        return null;
    }

    /// <summary>
    /// Gets only the outermost external contours (those with no parent).
    /// </summary>
    /// <param name="contours">The contour vectors.</param>
    /// <param name="hierarchy">The contour hierarchy matrix.</param>
    /// <returns>A new <see cref="VectorOfVectorOfPoint"/> containing only root-level contours. The caller is responsible for disposing it.</returns>
    /// <remarks>Only compatible with <see cref="RetrType.Tree"/> contour detection mode.</remarks>
    public static VectorOfVectorOfPoint GetExternalContours(VectorOfVectorOfPoint contours, int[,] hierarchy)
    {
        var result = new VectorOfVectorOfPoint();
        var vectorSize = contours.Size;
        for (var i = 0; i < vectorSize; i++)
        {
            if (hierarchy[i, EmguContour.HierarchyParent] != -1) continue;
            result.Push(contours[i]);
        }

        return result;
    }

    /// <summary>
    /// Gets all non-external (child/hole) contours.
    /// </summary>
    /// <param name="contours">The contour vectors.</param>
    /// <param name="hierarchy">The contour hierarchy matrix.</param>
    /// <returns>A new <see cref="VectorOfVectorOfPoint"/> containing only child contours. The caller is responsible for disposing it.</returns>
    public static VectorOfVectorOfPoint GetNegativeContours(VectorOfVectorOfPoint contours, int[,] hierarchy)
    {
        var result = new VectorOfVectorOfPoint();
        var vectorSize = contours.Size;
        for (var i = 0; i < vectorSize; i++)
        {
            if (hierarchy[i, EmguContour.HierarchyParent] == -1) continue;
            result.Push(contours[i]);
        }

        return result;
    }

    /// <summary>
    /// Gets both positive and negative contours grouped by their parent areas.
    /// </summary>
    /// <param name="contours">The contour vectors.</param>
    /// <param name="hierarchy">The contour hierarchy matrix.</param>
    /// <returns>An array of two lists: index 0 contains positive groups, index 1 contains negative groups.</returns>
    /// <remarks>Only compatible with <see cref="RetrType.Tree"/> contour detection mode.</remarks>
    public static List<VectorOfVectorOfPoint>[] GetContoursInGroups(VectorOfVectorOfPoint contours, int[,] hierarchy)
    {
        return [GetPositiveContoursInGroups(contours, hierarchy), GetNegativeContoursInGroups(contours, hierarchy)];
    }

    /// <summary>
    /// Gets positive (solid) contours grouped by their root parent, where each group contains a root contour and all its descendants.
    /// </summary>
    /// <param name="contours">The contour vectors.</param>
    /// <param name="hierarchy">The contour hierarchy matrix.</param>
    /// <returns>A list of <see cref="VectorOfVectorOfPoint"/> groups, one per external contour.</returns>
    /// <remarks>Only compatible with <see cref="RetrType.Tree"/> contour detection mode.</remarks>
    public static List<VectorOfVectorOfPoint> GetPositiveContoursInGroups(VectorOfVectorOfPoint contours, int[,] hierarchy)
    {
        var vectorSize = contours.Size;
        var result = new List<VectorOfVectorOfPoint>();
        var groupsByRoot = new Dictionary<int, VectorOfVectorOfPoint>();
        for (int i = 0; i < vectorSize; i++)
        {
            if (hierarchy[i, EmguContour.HierarchyParent] == -1)
            {
                var vec = new VectorOfVectorOfPoint(contours[i]);
                groupsByRoot.Add(i, vec);
                result.Add(vec);
            }
            else
            {
                int rootIndex = i;
                while (hierarchy[rootIndex, EmguContour.HierarchyParent] != -1)
                {
                    rootIndex = hierarchy[rootIndex, EmguContour.HierarchyParent];
                }

                groupsByRoot[rootIndex].Push(contours[i]);
            }
        }

        return result;
    }

    /// <summary>
    /// Gets negative (hole) contours grouped by their parent areas, where each group contains a hole contour and its nested descendants.
    /// </summary>
    /// <param name="contours">The contour vectors.</param>
    /// <param name="hierarchy">The contour hierarchy matrix.</param>
    /// <returns>A list of <see cref="VectorOfVectorOfPoint"/> groups, one per negative contour subtree.</returns>
    /// <remarks>Only compatible with <see cref="RetrType.Tree"/> contour detection mode.</remarks>
    public static List<VectorOfVectorOfPoint> GetNegativeContoursInGroups(VectorOfVectorOfPoint contours, int[,] hierarchy)
    {
        var result = new List<VectorOfVectorOfPoint>();
        var vectorSize = contours.Size;
        for (int i = 1; i < vectorSize; i++)
        {
            if (hierarchy[i, EmguContour.HierarchyParent] == -1) continue;
            var vec = new VectorOfVectorOfPoint(contours[i]);
            result.Add(vec);

            var contourId = i;
            for (i += 1; i < vectorSize && hierarchy[i, EmguContour.HierarchyParent] >= contourId; i++)
            {
                vec.Push(contours[i]);
            }

            i--;
        }

        return result;
    }

    /// <summary>
    /// Calculates the effective solid area for a grouped contour set (first contour area minus all subsequent contour areas).
    /// </summary>
    /// <param name="contours">A grouped contour vector where the first element is the outer contour and subsequent elements are inner holes.</param>
    /// <returns>The net solid area.</returns>
    public static double GetContourArea(VectorOfVectorOfPoint contours)
    {
        var vectorSize = contours.Size;
        if (vectorSize == 0) return 0;

        double result = CvInvoke.ContourArea(contours[0]);
        for (var i = 1; i < vectorSize; i++)
        {
            result -= CvInvoke.ContourArea(contours[i]);
        }
        return result;
    }

    /// <summary>
    /// Gets the largest individual contour area from a contour list.
    /// </summary>
    /// <param name="contours">The contour vectors to evaluate.</param>
    /// <returns>The largest contour area, or 0 if the list is empty.</returns>
    public static double GetLargestContourArea(VectorOfVectorOfPoint contours)
    {
        var vectorSize = contours.Size;
        if (vectorSize == 0) return 0;

        double result = 0;
        for (var i = 0; i < vectorSize; i++)
        {
            result = Math.Max(result, CvInvoke.ContourArea(contours[i]));
        }
        return result;
    }

    /// <summary>
    /// Gets the largest external contour area from a contour list, considering only root-level contours.
    /// </summary>
    /// <param name="contours">The contour vectors to evaluate.</param>
    /// <param name="hierarchy">The contour hierarchy matrix (comp or tree mode).</param>
    /// <returns>The largest external contour area, or 0 if the list is empty.</returns>
    public static double GetLargestContourArea(VectorOfVectorOfPoint contours, int[,] hierarchy)
    {
        var vectorSize = contours.Size;
        if (vectorSize == 0) return 0;

        double result = CvInvoke.ContourArea(contours[0]);
        for (var i = 1; i < vectorSize; i++)
        {
            if (hierarchy[i, EmguContour.HierarchyParent] != -1) continue;
            result = Math.Max(result, CvInvoke.ContourArea(contours[i]));
        }
        return result;
    }

    /// <summary>
    /// Calculates the effective solid area for each contour group.
    /// </summary>
    /// <param name="contours">A list of grouped contour vectors.</param>
    /// <param name="useParallel">If <see langword="true"/>, computes areas in parallel.</param>
    /// <returns>An array of areas with the same length as <paramref name="contours"/>.</returns>
    public static double[] GetContoursArea(List<VectorOfVectorOfPoint> contours, bool useParallel = false)
    {
        var result = new double[contours.Count];

        if (useParallel)
        {
            Parallel.For(0, contours.Count, i =>
            {
                result[i] = GetContourArea(contours[i]);
            });
        }
        else
        {
            for (var i = 0; i < contours.Count; i++)
            {
                result[i] = GetContourArea(contours[i]);
            }
        }

        return result;
    }

    /// <summary>
    /// Counts the number of overlapping pixels between two contour sets by rasterizing both and computing the bitwise AND.
    /// </summary>
    /// <param name="contour1">The first contour set.</param>
    /// <param name="contour2">The second contour set.</param>
    /// <returns>The number of intersecting pixels, or 0 if the bounding rectangles do not overlap.</returns>
    public static int ContoursIntersectingPixels(VectorOfVectorOfPoint contour1, VectorOfVectorOfPoint contour2)
    {
        if (contour1.Size == 0 || contour2.Size == 0) return 0;

        var contour1Rect = CvInvoke.BoundingRectangle(contour1[0]);
        var contour2Rect = CvInvoke.BoundingRectangle(contour2[0]);

        if (!contour1Rect.IntersectsWith(contour2Rect)) return 0;

        var totalRect = contour1Rect.Width * contour1Rect.Height <= contour2Rect.Width * contour2Rect.Height
            ? contour1Rect
            : contour2Rect;

        using var contour1Mat = EmguExtensions.InitMat(totalRect.Size);
        using var contour2Mat = EmguExtensions.InitMat(totalRect.Size);

        var inverseOffset = new Point(-totalRect.X, -totalRect.Y);
        CvInvoke.DrawContours(contour1Mat, contour1, -1, EmguExtensions.WhiteColor, -1, LineType.EightConnected, null, int.MaxValue, inverseOffset);
        CvInvoke.DrawContours(contour2Mat, contour2, -1, EmguExtensions.WhiteColor, -1, LineType.EightConnected, null, int.MaxValue, inverseOffset);

        CvInvoke.BitwiseAnd(contour1Mat, contour2Mat, contour1Mat);

        return CvInvoke.CountNonZero(contour1Mat);
    }

    /// <summary>
    /// Checks whether two contour sets have any overlapping pixels.
    /// </summary>
    /// <param name="contour1">The first contour set.</param>
    /// <param name="contour2">The second contour set.</param>
    /// <returns><see langword="true"/> if the contours share at least one pixel; otherwise <see langword="false"/>.</returns>
    public static bool ContoursIntersect(VectorOfVectorOfPoint contour1, VectorOfVectorOfPoint contour2)
    {
        return ContoursIntersectingPixels(contour1, contour2) > 0;
    }

    #endregion
}
