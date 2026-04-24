using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;

namespace EmguExtensions.Tests;

/// <summary>
/// Comprehensive tests for <see cref="EmguContours"/> and <see cref="EmguContourFamily"/>.
/// </summary>
public class UnitTestEmguContours
{
    #region Test data helpers

    /// <summary>
    /// Creates a 200×200 black image with a single filled white rectangle.
    /// </summary>
    private static Mat CreateSingleRectImage(Rectangle rect)
    {
        var mat = EmguExtensions.InitMat(new Size(200, 200));
        CvInvoke.Rectangle(mat, rect, EmguExtensions.WhiteColor, -1);
        return mat;
    }

    /// <summary>
    /// Creates a 200×200 black image with two separate filled white rectangles.
    /// </summary>
    private static Mat CreateTwoRectsImage(Rectangle rect1, Rectangle rect2)
    {
        var mat = EmguExtensions.InitMat(new Size(200, 200));
        CvInvoke.Rectangle(mat, rect1, EmguExtensions.WhiteColor, -1);
        CvInvoke.Rectangle(mat, rect2, EmguExtensions.WhiteColor, -1);
        return mat;
    }

    /// <summary>
    /// Creates a 200×200 black image with a filled white circle containing a black circle inside (donut).
    /// Returns an image that, when contours are found with RetrType.Tree, produces a parent/child hierarchy.
    /// </summary>
    private static Mat CreateDonutImage()
    {
        var mat = EmguExtensions.InitMat(new Size(200, 200));
        CvInvoke.Circle(mat, new Point(100, 100), 80, EmguExtensions.WhiteColor, -1);
        CvInvoke.Circle(mat, new Point(100, 100), 40, new MCvScalar(0), -1);
        return mat;
    }

    /// <summary>
    /// Creates a 300×300 black image with nested shapes:
    /// - Outer white rectangle
    /// - Inner black rectangle (hole)
    /// - Innermost white rectangle (solid inside hole)
    /// </summary>
    private static Mat CreateNestedImage()
    {
        var mat = EmguExtensions.InitMat(new Size(300, 300));
        CvInvoke.Rectangle(mat, new Rectangle(20, 20, 260, 260), EmguExtensions.WhiteColor, -1);
        CvInvoke.Rectangle(mat, new Rectangle(60, 60, 180, 180), new MCvScalar(0), -1);
        CvInvoke.Rectangle(mat, new Rectangle(100, 100, 100, 100), EmguExtensions.WhiteColor, -1);
        return mat;
    }

    /// <summary>
    /// Builds a hierarchy with known structure for testing without image processing:
    /// Contour 0: external (parent = -1)
    /// Contour 1: child of 0 (parent = 0)
    /// Contour 2: external (parent = -1)
    /// </summary>
    private static (Point[][] Points, int[,] Hierarchy) CreateManualHierarchy()
    {
        Point[][] points =
        [
            [new(10, 10), new(90, 10), new(90, 90), new(10, 90)],   // contour 0: outer square
            [new(30, 30), new(70, 30), new(70, 70), new(30, 70)],   // contour 1: inner square (child of 0)
            [new(110, 110), new(190, 110), new(190, 190), new(110, 190)]  // contour 2: separate outer square
        ];

        // [next_same_level, prev_same_level, first_child, parent]
        var hierarchy = new int[3, 4];
        // Contour 0: next=2, prev=-1, child=1, parent=-1
        hierarchy[0, 0] = 2;  hierarchy[0, 1] = -1; hierarchy[0, 2] = 1;  hierarchy[0, 3] = -1;
        // Contour 1: next=-1, prev=-1, child=-1, parent=0
        hierarchy[1, 0] = -1; hierarchy[1, 1] = -1; hierarchy[1, 2] = -1; hierarchy[1, 3] = 0;
        // Contour 2: next=-1, prev=0, child=-1, parent=-1
        hierarchy[2, 0] = -1; hierarchy[2, 1] = 0;  hierarchy[2, 2] = -1; hierarchy[2, 3] = -1;

        return (points, hierarchy);
    }

    #endregion

    #region Constructor - from Mat

    [Fact]
    public void Constructor_FromMat_SingleRect_FindsOneContour()
    {
        using var mat = CreateSingleRectImage(new Rectangle(50, 50, 100, 100));
        using var contours = new EmguContours(mat);

        Assert.Single(contours);
        Assert.False(contours.IsEmpty);
    }

    [Fact]
    public void Constructor_FromMat_TwoRects_FindsTwoContours()
    {
        using var mat = CreateTwoRectsImage(
            new Rectangle(10, 10, 50, 50),
            new Rectangle(140, 140, 50, 50));
        using var contours = new EmguContours(mat);

        Assert.Equal(2, contours.Count);
    }

    [Fact]
    public void Constructor_FromMat_EmptyImage_IsEmpty()
    {
        using var mat = EmguExtensions.InitMat(new Size(100, 100));
        using var contours = new EmguContours(mat);

        Assert.True(contours.IsEmpty);
        Assert.Empty(contours);
    }

    [Fact]
    public void Constructor_FromMat_Donut_FindsParentChild()
    {
        using var mat = CreateDonutImage();
        using var contours = new EmguContours(mat, RetrType.Tree);

        Assert.True(contours.Count >= 2);
        // With Tree mode, there should be families with children
        Assert.Single(contours.Families); // One external contour
        Assert.True(contours.Families[0].HasChildren);
    }

    [Fact]
    public void Constructor_FromMat_NestedImage_FindsThreeLevels()
    {
        using var mat = CreateNestedImage();
        using var contours = new EmguContours(mat, RetrType.Tree);

        Assert.True(contours.Count >= 3);
        // Outer → Hole → Inner solid
        Assert.Single(contours.Families);
        var root = contours.Families[0];
        Assert.True(root.HasChildren);
        Assert.True(root[0].HasChildren);
    }

    #endregion

    #region Constructor - from points and hierarchy

    [Fact]
    public void Constructor_FromPointsAndHierarchy_BuildsFamiliesCorrectly()
    {
        var (points, hierarchy) = CreateManualHierarchy();
        using var contours = new EmguContours(points, hierarchy);

        Assert.Equal(3, contours.Count);
        Assert.Equal(2, contours.Families.Count);
        Assert.Equal(2, contours.ExternalContoursCount);
    }

    [Fact]
    public void Constructor_FromPointsAndHierarchy_FamilyChildSetCorrectly()
    {
        var (points, hierarchy) = CreateManualHierarchy();
        using var contours = new EmguContours(points, hierarchy);

        var family0 = contours.Families[0]; // contour 0
        Assert.True(family0.HasChildren);
        Assert.Single(family0);
        Assert.Equal(1, family0[0].Index); // child is contour 1

        var family2 = contours.Families[1]; // contour 2
        Assert.False(family2.HasChildren);
    }

    #endregion

    #region Constructor - from VectorOfVectorOfPoint and hierarchy

    [Fact]
    public void Constructor_FromVector_LeaveOpenTrue_DoesNotDisposeVector()
    {
        var (points, hierarchy) = CreateManualHierarchy();
        var vec = new VectorOfVectorOfPoint(points);

        var contours = new EmguContours(vec, hierarchy, leaveOpen: true);
        contours.Dispose();

        // Vector should still be usable
        Assert.Equal(3, vec.Size);
        vec.Dispose();
    }

    #endregion

    #region Properties

    [Fact]
    public void IsEmpty_NoContours_ReturnsTrue()
    {
        using var mat = EmguExtensions.InitMat(new Size(50, 50));
        using var contours = new EmguContours(mat);

        Assert.True(contours.IsEmpty);
    }

    [Fact]
    public void IsEmpty_WithContours_ReturnsFalse()
    {
        using var mat = CreateSingleRectImage(new Rectangle(10, 10, 50, 50));
        using var contours = new EmguContours(mat);

        Assert.False(contours.IsEmpty);
    }

    [Fact]
    public void ExternalContoursCount_MatchesFamiliesCount()
    {
        using var mat = CreateTwoRectsImage(
            new Rectangle(10, 10, 50, 50),
            new Rectangle(140, 140, 50, 50));
        using var contours = new EmguContours(mat);

        Assert.Equal(contours.Families.Count, contours.ExternalContoursCount);
    }

    [Fact]
    public void ExternalContours_ReturnsRootContours()
    {
        var (points, hierarchy) = CreateManualHierarchy();
        using var contours = new EmguContours(points, hierarchy);

        var external = contours.ExternalContours.ToList();

        Assert.Equal(2, external.Count);
    }

    #endregion

    #region TotalSolidArea / MinSolidArea / MaxSolidArea

    [Fact]
    public void TotalSolidArea_EmptyContours_ReturnsZero()
    {
        using var mat = EmguExtensions.InitMat(new Size(50, 50));
        using var contours = new EmguContours(mat);

        Assert.Equal(0, contours.TotalSolidArea);
    }

    [Fact]
    public void TotalSolidArea_Donut_LessThanOuterArea()
    {
        using var mat = CreateDonutImage();
        using var contours = new EmguContours(mat, RetrType.Tree);

        var outerArea = contours[0].Area;
        var solidArea = contours.TotalSolidArea;

        Assert.True(solidArea > 0);
        Assert.True(solidArea < outerArea);
    }

    [Fact]
    public void TotalSolidArea_IsCachedOnSubsequentAccess()
    {
        using var mat = CreateSingleRectImage(new Rectangle(10, 10, 50, 50));
        using var contours = new EmguContours(mat, RetrType.Tree);

        var first = contours.TotalSolidArea;
        var second = contours.TotalSolidArea;

        Assert.Equal(first, second);
    }

    [Fact]
    public void MinSolidArea_EmptyContours_ReturnsZero()
    {
        using var mat = EmguExtensions.InitMat(new Size(50, 50));
        using var contours = new EmguContours(mat);

        Assert.Equal(0, contours.MinSolidArea);
    }

    [Fact]
    public void MaxSolidArea_EmptyContours_ReturnsZero()
    {
        using var mat = EmguExtensions.InitMat(new Size(50, 50));
        using var contours = new EmguContours(mat);

        Assert.Equal(0, contours.MaxSolidArea);
    }

    [Fact]
    public void MinMaxSolidArea_TwoRects_DifferentSizes()
    {
        using var mat = CreateTwoRectsImage(
            new Rectangle(10, 10, 30, 30),    // smaller
            new Rectangle(140, 140, 50, 50)); // larger
        using var contours = new EmguContours(mat, RetrType.Tree);

        Assert.True(contours.MinSolidArea > 0);
        Assert.True(contours.MaxSolidArea > 0);
        Assert.True(contours.MinSolidArea <= contours.MaxSolidArea);
    }

    #endregion

    #region Indexers

    [Fact]
    public void Indexer_Int_ReturnsCorrectContour()
    {
        var (points, hierarchy) = CreateManualHierarchy();
        using var contours = new EmguContours(points, hierarchy);

        var c0 = contours[0];
        var c1 = contours[1];
        var c2 = contours[2];

        Assert.Equal(4, c0.Count);
        Assert.Equal(4, c1.Count);
        Assert.Equal(4, c2.Count);
    }

    [Fact]
    public void Indexer_MultipleTypes_AllReturnSameContour()
    {
        var (points, hierarchy) = CreateManualHierarchy();
        using var contours = new EmguContours(points, hierarchy);

        Assert.Same(contours[0], contours[(byte)0]);
        Assert.Same(contours[0], contours[(short)0]);
        Assert.Same(contours[0], contours[(ushort)0]);
        Assert.Same(contours[0], contours[(uint)0]);
        Assert.Same(contours[0], contours[(long)0]);
        Assert.Same(contours[0], contours[(ulong)0]);
        Assert.Same(contours[0], contours[(sbyte)0]);
    }

    [Fact]
    public void Indexer_Hierarchy_ReturnsCorrectValues()
    {
        var (points, hierarchy) = CreateManualHierarchy();
        using var contours = new EmguContours(points, hierarchy);

        // Contour 0: parent = -1 (root)
        Assert.Equal(-1, contours[0, EmguContour.HierarchyParent]);
        // Contour 1: parent = 0
        Assert.Equal(0, contours[1, EmguContour.HierarchyParent]);
        // Contour 2: parent = -1 (root)
        Assert.Equal(-1, contours[2, EmguContour.HierarchyParent]);
    }

    [Fact]
    public void Indexer_HierarchyMultipleTypes_AllReturnSameValue()
    {
        var (points, hierarchy) = CreateManualHierarchy();
        using var contours = new EmguContours(points, hierarchy);

        int expected = contours[0, EmguContour.HierarchyParent];
        Assert.Equal(expected, contours[(byte)0, EmguContour.HierarchyParent]);
        Assert.Equal(expected, contours[(short)0, EmguContour.HierarchyParent]);
        Assert.Equal(expected, contours[(ushort)0, EmguContour.HierarchyParent]);
        Assert.Equal(expected, contours[(uint)0, EmguContour.HierarchyParent]);
        Assert.Equal(expected, contours[(long)0, EmguContour.HierarchyParent]);
        Assert.Equal(expected, contours[(ulong)0, EmguContour.HierarchyParent]);
        Assert.Equal(expected, contours[(sbyte)0, EmguContour.HierarchyParent]);
    }

    #endregion

    #region IReadOnlyList

    [Fact]
    public void GetEnumerator_IteratesAllContours()
    {
        var (points, hierarchy) = CreateManualHierarchy();
        using var contours = new EmguContours(points, hierarchy);

        var list = contours.ToList();

        Assert.Equal(3, list.Count);
    }

    [Fact]
    public void Count_MatchesVectorSize()
    {
        var (points, hierarchy) = CreateManualHierarchy();
        using var contours = new EmguContours(points, hierarchy);

        Assert.Equal(3, contours.Count);
        Assert.Equal(contours.Vector.Size, contours.Count);
    }

    #endregion

    #region CalculateCentroidDistances

    [Fact]
    public void CalculateCentroidDistances_ExcludeOwn_CorrectLength()
    {
        var (points, hierarchy) = CreateManualHierarchy();
        using var contours = new EmguContours(points, hierarchy);

        var distances = contours.CalculateCentroidDistances(includeOwn: false);

        Assert.Equal(3, distances.Length);
        Assert.Equal(2, distances[0].Length); // 3 contours - 1 = 2
        Assert.Equal(2, distances[1].Length);
        Assert.Equal(2, distances[2].Length);
    }

    [Fact]
    public void CalculateCentroidDistances_IncludeOwn_CorrectLength()
    {
        var (points, hierarchy) = CreateManualHierarchy();
        using var contours = new EmguContours(points, hierarchy);

        var distances = contours.CalculateCentroidDistances(includeOwn: true);

        Assert.Equal(3, distances.Length);
        Assert.Equal(3, distances[0].Length);
        // Own distance should be 0
        Assert.Contains(distances[0], d => d is { Index: 0, Distance: 0 });
    }

    [Fact]
    public void CalculateCentroidDistances_SortedByDistance()
    {
        var (points, hierarchy) = CreateManualHierarchy();
        using var contours = new EmguContours(points, hierarchy);

        var distances = contours.CalculateCentroidDistances(sortByDistance: true);

        for (int i = 0; i < distances.Length; i++)
        {
            for (int j = 1; j < distances[i].Length; j++)
            {
                Assert.True(distances[i][j - 1].Distance <= distances[i][j].Distance);
            }
        }
    }

    [Fact]
    public void CalculateCentroidDistances_SymmetricDistances()
    {
        var (points, hierarchy) = CreateManualHierarchy();
        using var contours = new EmguContours(points, hierarchy);

        var distances = contours.CalculateCentroidDistances(includeOwn: true, sortByDistance: false);

        // Distance from 0→1 should equal distance from 1→0
        var d01 = distances[0].First(d => d.Index == 1).Distance;
        var d10 = distances[1].First(d => d.Index == 0).Distance;
        Assert.Equal(d01, d10, 1e-10);
    }

    #endregion

    #region Clone

    [Fact]
    public void Clone_ProducesEqualCollection()
    {
        var (points, hierarchy) = CreateManualHierarchy();
        using var original = new EmguContours(points, hierarchy);

        using var clone = original.Clone();

        Assert.Equal(original.Count, clone.Count);
        Assert.Equal(original.Families.Count, clone.Families.Count);
    }

    [Fact]
    public void Clone_IndependentData()
    {
        using var mat = CreateSingleRectImage(new Rectangle(50, 50, 100, 100));
        using var original = new EmguContours(mat);
        using var clone = original.Clone();

        Assert.NotSame(original.Vector, clone.Vector);
    }

    #endregion

    #region Static GetContoursInside

    [Fact]
    public void GetContoursInside_PointInsideContour_ReturnsContour()
    {
        using var mat = CreateSingleRectImage(new Rectangle(50, 50, 100, 100));
        using var contours = new EmguContours(mat, RetrType.Tree);

        using var result = contours.GetContoursInside(new Point(100, 100));

        Assert.True(result.Size > 0);
    }

    [Fact]
    public void GetContoursInside_PointOutside_ReturnsEmpty()
    {
        using var mat = CreateSingleRectImage(new Rectangle(50, 50, 100, 100));
        using var contours = new EmguContours(mat, RetrType.Tree);

        using var result = contours.GetContoursInside(new Point(5, 5));

        Assert.Equal(0, result.Size);
    }

    [Fact]
    public void GetContoursInside_Static_PointInside_ReturnsContour()
    {
        using var mat = CreateSingleRectImage(new Rectangle(50, 50, 100, 100));
        using var contours = new EmguContours(mat, RetrType.Tree);

        using var result = EmguContours.GetContoursInside(
            contours.Vector, contours.Hierarchy,
            new Point(100, 100));

        Assert.True(result.Size > 0);
    }

    #endregion

    #region Static GetContourInside

    [Fact]
    public void GetContourInside_PointInside_ReturnsContourAndIndex()
    {
        using var mat = CreateSingleRectImage(new Rectangle(50, 50, 100, 100));
        using var contours = new EmguContours(mat, RetrType.Tree);

        var result = EmguContours.GetContourInside(
            contours.Vector, new Point(100, 100), out int index);

        Assert.NotNull(result);
        Assert.True(index >= 0);
    }

    [Fact]
    public void GetContourInside_PointOutside_ReturnsNullAndNegativeIndex()
    {
        using var mat = CreateSingleRectImage(new Rectangle(50, 50, 100, 100));
        using var contours = new EmguContours(mat, RetrType.Tree);

        var result = EmguContours.GetContourInside(
            contours.Vector, new Point(5, 5), out int index);

        Assert.Null(result);
        Assert.Equal(-1, index);
    }

    #endregion

    #region Static GetExternalContours

    [Fact]
    public void GetExternalContours_Donut_ReturnsOnlyOuter()
    {
        using var mat = CreateDonutImage();
        using var contours = new EmguContours(mat, RetrType.Tree);

        using var external = EmguContours.GetExternalContours(contours.Vector, contours.Hierarchy);

        Assert.Equal(1, external.Size);
    }

    [Fact]
    public void GetExternalContours_TwoSeparateRects_ReturnsBoth()
    {
        using var mat = CreateTwoRectsImage(
            new Rectangle(10, 10, 50, 50),
            new Rectangle(140, 140, 50, 50));
        using var contours = new EmguContours(mat, RetrType.Tree);

        using var external = EmguContours.GetExternalContours(contours.Vector, contours.Hierarchy);

        Assert.Equal(2, external.Size);
    }

    #endregion

    #region Static GetNegativeContours

    [Fact]
    public void GetNegativeContours_Donut_ReturnsHoleContour()
    {
        using var mat = CreateDonutImage();
        using var contours = new EmguContours(mat, RetrType.Tree);

        using var negative = EmguContours.GetNegativeContours(contours.Vector, contours.Hierarchy);

        Assert.True(negative.Size >= 1);
    }

    [Fact]
    public void GetNegativeContours_SingleRect_ReturnsEmpty()
    {
        using var mat = CreateSingleRectImage(new Rectangle(50, 50, 100, 100));
        using var contours = new EmguContours(mat, RetrType.Tree);

        using var negative = EmguContours.GetNegativeContours(contours.Vector, contours.Hierarchy);

        Assert.Equal(0, negative.Size);
    }

    #endregion

    #region Static GetContoursInGroups

    [Fact]
    public void GetContoursInGroups_ReturnsPositiveAndNegativeGroups()
    {
        using var mat = CreateDonutImage();
        using var contours = new EmguContours(mat, RetrType.Tree);

        var groups = EmguContours.GetContoursInGroups(contours.Vector, contours.Hierarchy);

        Assert.Equal(2, groups.Length);

        // Positive groups
        Assert.True(groups[0].Count > 0);

        // Negative groups (donut has at least one hole)
        Assert.True(groups[1].Count >= 1);

        // Cleanup
        foreach (var g in groups[0]) g.Dispose();
        foreach (var g in groups[1]) g.Dispose();
    }

    #endregion

    #region Static GetContourArea

    [Fact]
    public void GetContourArea_EmptyVector_ReturnsZero()
    {
        using var vec = new VectorOfVectorOfPoint();

        var area = EmguContours.GetContourArea(vec);

        Assert.Equal(0, area);
    }

    [Fact]
    public void GetContourArea_SingleContour_ReturnsItsArea()
    {
        Point[] square = [new(10, 10), new(90, 10), new(90, 90), new(10, 90)];
        using var vec = new VectorOfVectorOfPoint(new VectorOfPoint(square));

        var area = EmguContours.GetContourArea(vec);

        Assert.Equal(6400, area);
    }

    [Fact]
    public void GetContourArea_WithHole_SubtractsHoleArea()
    {
        Point[] outer = [new(0, 0), new(100, 0), new(100, 100), new(0, 100)];
        Point[] inner = [new(20, 20), new(80, 20), new(80, 80), new(20, 80)];
        using var vec = new VectorOfVectorOfPoint();
        vec.Push(new VectorOfPoint(outer));
        vec.Push(new VectorOfPoint(inner));

        var area = EmguContours.GetContourArea(vec);

        // 10000 - 3600 = 6400
        Assert.Equal(6400, area);
    }

    #endregion

    #region Static GetLargestContourArea

    [Fact]
    public void GetLargestContourArea_EmptyVector_ReturnsZero()
    {
        using var vec = new VectorOfVectorOfPoint();

        Assert.Equal(0, EmguContours.GetLargestContourArea(vec));
    }

    [Fact]
    public void GetLargestContourArea_TwoContours_ReturnsLargest()
    {
        Point[] small = [new(0, 0), new(10, 0), new(10, 10), new(0, 10)];
        Point[] large = [new(0, 0), new(50, 0), new(50, 50), new(0, 50)];
        using var vec = new VectorOfVectorOfPoint();
        vec.Push(new VectorOfPoint(small));
        vec.Push(new VectorOfPoint(large));

        var area = EmguContours.GetLargestContourArea(vec);

        Assert.Equal(2500, area);
    }

    [Fact]
    public void GetLargestContourArea_WithHierarchy_ReturnsLargestExternal()
    {
        using var mat = CreateTwoRectsImage(
            new Rectangle(10, 10, 30, 30),    // smaller
            new Rectangle(140, 140, 50, 50)); // larger
        using var contours = new EmguContours(mat, RetrType.Tree);

        var largest = EmguContours.GetLargestContourArea(contours.Vector, contours.Hierarchy);

        // Should be approximately 50*50 = 2500
        Assert.True(largest >= 2400);
    }

    #endregion

    #region Static GetContoursArea

    [Fact]
    public void GetContoursArea_Sequential_ReturnsCorrectAreas()
    {
        Point[] square1 = [new(0, 0), new(10, 0), new(10, 10), new(0, 10)];
        Point[] square2 = [new(0, 0), new(20, 0), new(20, 20), new(0, 20)];
        var groups = new List<VectorOfVectorOfPoint>
        {
            new(new VectorOfPoint(square1)),
            new(new VectorOfPoint(square2))
        };

        var areas = EmguContours.GetContoursArea(groups);

        Assert.Equal(2, areas.Length);
        Assert.Equal(100, areas[0]);
        Assert.Equal(400, areas[1]);

        foreach (var g in groups) g.Dispose();
    }

    [Fact]
    public void GetContoursArea_Parallel_ReturnsSameAsSequential()
    {
        Point[] square1 = [new(0, 0), new(10, 0), new(10, 10), new(0, 10)];
        Point[] square2 = [new(0, 0), new(20, 0), new(20, 20), new(0, 20)];
        var groups = new List<VectorOfVectorOfPoint>
        {
            new(new VectorOfPoint(square1)),
            new(new VectorOfPoint(square2))
        };

        var sequential = EmguContours.GetContoursArea(groups, useParallel: false);
        var parallel = EmguContours.GetContoursArea(groups, useParallel: true);

        Assert.Equal(sequential, parallel);

        foreach (var g in groups) g.Dispose();
    }

    #endregion

    #region Static ContoursIntersect / ContoursIntersectingPixels

    [Fact]
    public void ContoursIntersect_OverlappingContours_ReturnsTrue()
    {
        Point[] rect1 = [new(0, 0), new(50, 0), new(50, 50), new(0, 50)];
        Point[] rect2 = [new(25, 25), new(75, 25), new(75, 75), new(25, 75)];
        using var vec1 = new VectorOfVectorOfPoint(new VectorOfPoint(rect1));
        using var vec2 = new VectorOfVectorOfPoint(new VectorOfPoint(rect2));

        Assert.True(EmguContours.ContoursIntersect(vec1, vec2));
    }

    [Fact]
    public void ContoursIntersect_NonOverlappingContours_ReturnsFalse()
    {
        Point[] rect1 = [new(0, 0), new(10, 0), new(10, 10), new(0, 10)];
        Point[] rect2 = [new(50, 50), new(60, 50), new(60, 60), new(50, 60)];
        using var vec1 = new VectorOfVectorOfPoint(new VectorOfPoint(rect1));
        using var vec2 = new VectorOfVectorOfPoint(new VectorOfPoint(rect2));

        Assert.False(EmguContours.ContoursIntersect(vec1, vec2));
    }

    [Fact]
    public void ContoursIntersectingPixels_OverlappingContours_ReturnsPositiveCount()
    {
        Point[] rect1 = [new(0, 0), new(50, 0), new(50, 50), new(0, 50)];
        Point[] rect2 = [new(25, 25), new(75, 25), new(75, 75), new(25, 75)];
        using var vec1 = new VectorOfVectorOfPoint(new VectorOfPoint(rect1));
        using var vec2 = new VectorOfVectorOfPoint(new VectorOfPoint(rect2));

        var pixels = EmguContours.ContoursIntersectingPixels(vec1, vec2);

        Assert.True(pixels > 0);
    }

    [Fact]
    public void ContoursIntersectingPixels_NonOverlapping_ReturnsZero()
    {
        Point[] rect1 = [new(0, 0), new(10, 0), new(10, 10), new(0, 10)];
        Point[] rect2 = [new(50, 50), new(60, 50), new(60, 60), new(50, 60)];
        using var vec1 = new VectorOfVectorOfPoint(new VectorOfPoint(rect1));
        using var vec2 = new VectorOfVectorOfPoint(new VectorOfPoint(rect2));

        Assert.Equal(0, EmguContours.ContoursIntersectingPixels(vec1, vec2));
    }

    #endregion
}

/// <summary>
/// Comprehensive tests for <see cref="EmguContourFamily"/>.
/// </summary>
public class UnitTestEmguContourFamily
{
    #region Test data helpers

    /// <summary>
    /// Creates a donut image (outer circle with inner hole) and returns the EmguContours.
    /// Caller must dispose.
    /// </summary>
    private static EmguContours CreateDonutContours()
    {
        var mat = EmguExtensions.InitMat(new Size(200, 200));
        CvInvoke.Circle(mat, new Point(100, 100), 80, EmguExtensions.WhiteColor, -1);
        CvInvoke.Circle(mat, new Point(100, 100), 40, new MCvScalar(0), -1);
        var contours = new EmguContours(mat, RetrType.Tree);
        mat.Dispose();
        return contours;
    }

    /// <summary>
    /// Creates a 3-level nested image and returns the EmguContours.
    /// Outer white → Inner black hole → Innermost white solid.
    /// Caller must dispose.
    /// </summary>
    private static EmguContours CreateNestedContours()
    {
        var mat = EmguExtensions.InitMat(new Size(300, 300));
        CvInvoke.Rectangle(mat, new Rectangle(20, 20, 260, 260), EmguExtensions.WhiteColor, -1);
        CvInvoke.Rectangle(mat, new Rectangle(60, 60, 180, 180), new MCvScalar(0), -1);
        CvInvoke.Rectangle(mat, new Rectangle(100, 100, 100, 100), EmguExtensions.WhiteColor, -1);
        var contours = new EmguContours(mat, RetrType.Tree);
        mat.Dispose();
        return contours;
    }

    /// <summary>
    /// Creates a manually constructed hierarchy for controlled testing.
    /// </summary>
    private static EmguContours CreateManualContours()
    {
        Point[][] points =
        [
            [new(10, 10), new(90, 10), new(90, 90), new(10, 90)],
            [new(30, 30), new(70, 30), new(70, 70), new(30, 70)],
            [new(40, 40), new(60, 40), new(60, 60), new(40, 60)]
        ];

        // 0: root, 1: child of 0, 2: child of 1
        var hierarchy = new int[3, 4];
        hierarchy[0, 0] = -1; hierarchy[0, 1] = -1; hierarchy[0, 2] = 1;  hierarchy[0, 3] = -1;
        hierarchy[1, 0] = -1; hierarchy[1, 1] = -1; hierarchy[1, 2] = 2;  hierarchy[1, 3] = 0;
        hierarchy[2, 0] = -1; hierarchy[2, 1] = -1; hierarchy[2, 2] = -1; hierarchy[2, 3] = 1;

        return new EmguContours(points, hierarchy);
    }

    #endregion

    #region Properties

    [Fact]
    public void Index_ReturnsCorrectContourIndex()
    {
        using var contours = CreateManualContours();

        Assert.Equal(0, contours.Families[0].Index);
    }

    [Fact]
    public void Depth_Root_IsZero()
    {
        using var contours = CreateManualContours();

        Assert.Equal(0, contours.Families[0].Depth);
    }

    [Fact]
    public void Depth_Child_IsOne()
    {
        using var contours = CreateManualContours();

        Assert.Equal(1, contours.Families[0][0].Depth);
    }

    [Fact]
    public void Depth_Grandchild_IsTwo()
    {
        using var contours = CreateManualContours();

        Assert.Equal(2, contours.Families[0][0][0].Depth);
    }

    [Fact]
    public void Self_ReturnsEmguContour()
    {
        using var contours = CreateManualContours();

        var family = contours.Families[0];

        Assert.NotNull(family.Self);
        Assert.IsType<EmguContour>(family.Self);
    }

    [Fact]
    public void Parent_Root_IsNull()
    {
        using var contours = CreateManualContours();

        Assert.Null(contours.Families[0].Parent);
    }

    [Fact]
    public void Parent_Child_ReturnsParent()
    {
        using var contours = CreateManualContours();

        var child = contours.Families[0][0];

        Assert.NotNull(child.Parent);
        Assert.Same(contours.Families[0], child.Parent);
    }

    [Fact]
    public void ParentIndex_Root_ReturnsMinusOne()
    {
        using var contours = CreateManualContours();

        Assert.Equal(-1, contours.Families[0].ParentIndex);
    }

    [Fact]
    public void ParentIndex_Child_ReturnsParentsIndex()
    {
        using var contours = CreateManualContours();

        Assert.Equal(0, contours.Families[0][0].ParentIndex);
    }

    #endregion

    #region IsExternal / IsPositive / IsNegative

    [Fact]
    public void IsExternal_Root_ReturnsTrue()
    {
        using var contours = CreateManualContours();

        Assert.True(contours.Families[0].IsExternal);
    }

    [Fact]
    public void IsExternal_Child_ReturnsFalse()
    {
        using var contours = CreateManualContours();

        Assert.False(contours.Families[0][0].IsExternal);
    }

    [Fact]
    public void IsPositive_EvenDepth_ReturnsTrue()
    {
        using var contours = CreateManualContours();

        Assert.True(contours.Families[0].IsPositive);         // depth 0
        Assert.True(contours.Families[0][0][0].IsPositive);   // depth 2
    }

    [Fact]
    public void IsNegative_OddDepth_ReturnsTrue()
    {
        using var contours = CreateManualContours();

        Assert.True(contours.Families[0][0].IsNegative);  // depth 1
    }

    [Fact]
    public void IsPositive_IsNegative_AreMutuallyExclusive()
    {
        using var contours = CreateManualContours();

        foreach (var family in contours.Families[0].TraverseTree())
        {
            Assert.NotEqual(family.IsPositive, family.IsNegative);
        }
    }

    #endregion

    #region HasChildren / Count

    [Fact]
    public void HasChildren_WithChildren_ReturnsTrue()
    {
        using var contours = CreateManualContours();

        Assert.True(contours.Families[0].HasChildren);
    }

    [Fact]
    public void HasChildren_Leaf_ReturnsFalse()
    {
        using var contours = CreateManualContours();

        Assert.False(contours.Families[0][0][0].HasChildren);
    }

    [Fact]
    public void Count_ReturnsChildrenCount()
    {
        using var contours = CreateManualContours();

        Assert.Single(contours.Families[0]); // one child
        Assert.Single(contours.Families[0][0]); // one grandchild
        Assert.Empty(contours.Families[0][0][0]); // leaf
    }

    #endregion

    #region Root

    [Fact]
    public void Root_RootNode_ReturnsSelf()
    {
        using var contours = CreateManualContours();

        var root = contours.Families[0];

        Assert.Same(root, root.Root);
    }

    [Fact]
    public void Root_ChildNode_ReturnsRoot()
    {
        using var contours = CreateManualContours();

        var child = contours.Families[0][0];

        Assert.Same(contours.Families[0], child.Root);
    }

    [Fact]
    public void Root_GrandchildNode_ReturnsRoot()
    {
        using var contours = CreateManualContours();

        var grandchild = contours.Families[0][0][0];

        Assert.Same(contours.Families[0], grandchild.Root);
    }

    #endregion

    #region TotalSolidArea

    [Fact]
    public void TotalSolidArea_SolidContour_EqualsArea()
    {
        Point[][] points = [[new(0, 0), new(100, 0), new(100, 100), new(0, 100)]];
        var hierarchy = new int[1, 4];
        hierarchy[0, 0] = -1; hierarchy[0, 1] = -1; hierarchy[0, 2] = -1; hierarchy[0, 3] = -1;

        using var contours = new EmguContours(points, hierarchy);
        var family = contours.Families[0];

        Assert.Equal(family.Self.Area, family.TotalSolidArea);
    }

    [Fact]
    public void TotalSolidArea_WithHole_SubtractsHoleArea()
    {
        using var contours = CreateManualContours();
        var root = contours.Families[0];

        var rootArea = root.Self.Area;
        var solidArea = root.TotalSolidArea;

        // Solid area should be less than root area because of the hole,
        // but greater than 0 because of the solid inside the hole
        Assert.True(solidArea < rootArea);
        Assert.True(solidArea > 0);
    }

    [Fact]
    public void TotalSolidArea_IsCached()
    {
        using var contours = CreateManualContours();
        var root = contours.Families[0];

        var first = root.TotalSolidArea;
        var second = root.TotalSolidArea;

        Assert.Equal(first, second);
    }

    #endregion

    #region TraverseTree

    [Fact]
    public void TraverseTree_ReturnsAllNodesIncludingSelf()
    {
        using var contours = CreateManualContours();
        var root = contours.Families[0];

        var nodes = root.TraverseTree().ToList();

        Assert.Equal(3, nodes.Count);
    }

    [Fact]
    public void TraverseTree_BreadthFirst_RootIsFirst()
    {
        using var contours = CreateManualContours();
        var root = contours.Families[0];

        var nodes = root.TraverseTree().ToList();

        Assert.Same(root, nodes[0]);
    }

    [Fact]
    public void TraverseTree_LeafNode_ReturnsSingleElement()
    {
        using var contours = CreateManualContours();
        var leaf = contours.Families[0][0][0];

        var nodes = leaf.TraverseTree().ToList();

        Assert.Single(nodes);
        Assert.Same(leaf, nodes[0]);
    }

    [Fact]
    public void TraverseTreeAsEmguContour_ReturnsContourObjects()
    {
        using var contours = CreateManualContours();
        var root = contours.Families[0];

        var emguContours = root.TraverseTreeAsEmguContour().ToList();

        Assert.Equal(3, emguContours.Count);
        Assert.All(emguContours, c => Assert.IsType<EmguContour>(c));
    }

    #endregion

    #region ToVectorOfVectorOfPoint / ToEmguContourArray

    [Fact]
    public void ToVectorOfVectorOfPoint_ReturnsAllContourVectors()
    {
        using var contours = CreateManualContours();
        var root = contours.Families[0];

        using var vec = root.ToVectorOfVectorOfPoint();

        Assert.Equal(3, vec.Size);
    }

    [Fact]
    public void ToEmguContourArray_ReturnsAllContours()
    {
        using var contours = CreateManualContours();
        var root = contours.Families[0];

        var array = root.ToEmguContourArray();

        Assert.Equal(3, array.Length);
    }

    #endregion

    #region IReadOnlyList / GetEnumerator

    [Fact]
    public void GetEnumerator_IteratesChildren()
    {
        using var contours = CreateManualContours();
        var root = contours.Families[0];

        var children = root.ToList();

        Assert.Single(children);
        Assert.Equal(1, children[0].Index);
    }

    [Fact]
    public void Indexer_ReturnsCorrectChild()
    {
        using var contours = CreateManualContours();
        var root = contours.Families[0];

        var child = root[0];

        Assert.Equal(1, child.Depth);
    }

    #endregion
}
