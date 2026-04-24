using System.Drawing;
using Emgu.CV;
using Emgu.CV.Structure;

namespace EmguExtensions.Tests;

public class UnitTestMatRoi
{
    // 100×80 mat: top half black (0), bottom half white (255).
    private static Mat CreateMat(int width = 100, int height = 80)
    {
        var mat = EmguExtensions.InitMat(new Size(width, height));
        CvInvoke.Rectangle(mat, new Rectangle(0, height / 2, width, height - height / 2),
            EmguExtensions.WhiteColor, -1);
        return mat;
    }

    // Returns the bytes of a Mat as a contiguous copy (safe for submatrix views).
    private static byte[] ToContiguousBytes(Mat mat)
    {
        using var copy = mat.Clone();
        return copy.ToArray();
    }

    #region Constructor(Mat, Rectangle, bool)

    [Fact]
    public void Constructor_ValidMatAndRoi_SetsAllProperties()
    {
        using var mat = CreateMat();
        var roi = new Rectangle(10, 20, 50, 30);

        using var matRoi = new MatRoi(mat, roi);

        Assert.Same(mat, matRoi.SourceMat);
        Assert.Equal(roi, matRoi.Roi);
        Assert.Equal(roi.Size, matRoi.RoiMat.Size);
        Assert.False(matRoi.RoiMat.IsEmpty);
    }

    [Fact]
    public void Constructor_NullMat_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() => new MatRoi((Mat)null!, new Rectangle(0, 0, 10, 10)));
    }

    [Fact]
    public void Constructor_EmptySourceMat_RoiMatIsEmptyAndRoiIsEmpty()
    {
        using var empty = new Mat();

        using var matRoi = new MatRoi(empty, new Rectangle(0, 0, 10, 10));

        Assert.Equal(Rectangle.Empty, matRoi.Roi);
        Assert.True(matRoi.RoiMat.IsEmpty);
    }

    [Fact]
    public void Constructor_RectangleEmptyRoi_RoiMatIsEmptyAndRoiIsEmpty()
    {
        using var mat = CreateMat();

        using var matRoi = new MatRoi(mat, Rectangle.Empty);

        Assert.Equal(Rectangle.Empty, matRoi.Roi);
        Assert.True(matRoi.RoiMat.IsEmpty);
    }

    [Fact]
    public void Constructor_RoiFullyOutsideBounds_RoiMatIsEmptyAndRoiIsEmpty()
    {
        using var mat = CreateMat(100, 80);

        using var matRoi = new MatRoi(mat, new Rectangle(200, 200, 50, 50));

        Assert.Equal(Rectangle.Empty, matRoi.Roi);
        Assert.True(matRoi.RoiMat.IsEmpty);
    }

    [Fact]
    public void Constructor_RoiPartiallyOutsideBounds_RoiClampedToSourceBounds()
    {
        using var mat = CreateMat(100, 80);
        // Right edge: 80+40=120 > 100 → clamped to 100. Bottom: 60+40=100 > 80 → clamped to 80.
        using var matRoi = new MatRoi(mat, new Rectangle(80, 60, 40, 40));

        Assert.Equal(new Rectangle(80, 60, 20, 20), matRoi.Roi);
        Assert.Equal(new Size(20, 20), matRoi.RoiMat.Size);
    }

    [Fact]
    public void Constructor_SinglePixelRoi_ReturnsOneByOneMat()
    {
        using var mat = CreateMat();

        using var matRoi = new MatRoi(mat, new Rectangle(5, 5, 1, 1));

        Assert.Equal(new Rectangle(5, 5, 1, 1), matRoi.Roi);
        Assert.Equal(new Size(1, 1), matRoi.RoiMat.Size);
    }

    [Fact]
    public void Constructor_RoiMatchesFullSourceSize_SetsRoiCorrectly()
    {
        using var mat = CreateMat(100, 80);
        var fullRoi = new Rectangle(0, 0, 100, 80);

        using var matRoi = new MatRoi(mat, fullRoi);

        Assert.Equal(fullRoi, matRoi.Roi);
        Assert.Equal(mat.Size, matRoi.RoiMat.Size);
    }

    #endregion

    #region Constructor(Mat, Rectangle, int, int, int, int, bool)

    [Fact]
    public void Constructor_NullMatWithPerSidePadding_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new MatRoi(null!, new Rectangle(0, 0, 10, 10), 5, 5, 5, 5));
    }

    [Fact]
    public void Constructor_WithPadding_RoiExpandedByPaddingAmount()
    {
        using var mat = CreateMat(100, 80);
        var inner = new Rectangle(20, 20, 30, 20);

        using var matRoi = new MatRoi(mat, inner, padLeft: 5, padTop: 5, padRight: 5, padBottom: 5);

        // Expected: x=15, y=15, w=40, h=30
        Assert.Equal(new Rectangle(15, 15, 40, 30), matRoi.Roi);
        Assert.Equal(new Size(40, 30), matRoi.RoiMat.Size);
    }

    [Fact]
    public void Constructor_WithPaddingExceedingSourceBounds_RoiClampedToSource()
    {
        using var mat = CreateMat(100, 80);
        // ROI near top-left; padding of 10 would go negative → clamped to 0.
        var inner = new Rectangle(5, 5, 30, 20);

        using var matRoi = new MatRoi(mat, inner, padLeft: 10, padTop: 10, padRight: 10, padBottom: 10);

        // x: max(0, 5-10)=0, y: max(0, 5-10)=0
        // right: min(100, 35+10)=45, bottom: min(80, 25+10)=35
        Assert.Equal(new Rectangle(0, 0, 45, 35), matRoi.Roi);
    }

    [Fact]
    public void Constructor_ZeroPadding_ProducesSameRoiAsNoPaddingOverload()
    {
        using var mat = CreateMat(100, 80);
        var roi = new Rectangle(10, 10, 50, 40);

        using var noPad = new MatRoi(mat, roi);
        using var zeroPad = new MatRoi(mat, roi, 0, 0, 0, 0);

        Assert.Equal(noPad.Roi, zeroPad.Roi);
        Assert.Equal(noPad.RoiMat.Size, zeroPad.RoiMat.Size);
    }

    #endregion

    #region Constructor(Mat, Rectangle, Size, bool)

    [Fact]
    public void Constructor_NullMatWithSizePadding_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new MatRoi(null!, new Rectangle(0, 0, 10, 10), new Size(5, 5)));
    }

    [Fact]
    public void Constructor_SizePadding_EquivalentToPerSidePaddingWithSameValues()
    {
        using var mat = CreateMat(100, 80);
        var roi = new Rectangle(20, 20, 30, 20);

        using var perSide = new MatRoi(mat, roi, padLeft: 5, padTop: 5, padRight: 5, padBottom: 5);
        using var sizePad = new MatRoi(mat, roi, new Size(5, 5));

        Assert.Equal(perSide.Roi, sizePad.Roi);
        Assert.Equal(perSide.RoiMat.Size, sizePad.RoiMat.Size);
    }

    #endregion

    #region Constructor(MatRoi, Rectangle)

    [Fact]
    public void Constructor_FromMatRoi_UsesSourceMatAndSetsLeaveOpenTrue()
    {
        using var mat = CreateMat(100, 80);
        using var outer = new MatRoi(mat, new Rectangle(0, 0, 80, 60));

        using var inner = new MatRoi(outer, new Rectangle(10, 10, 30, 20));

        Assert.Same(mat, inner.SourceMat);
        Assert.True(inner.LeaveOpen);
    }

    [Fact]
    public void Constructor_FromMatRoi_InnerDisposeDoesNotDisposeSourceMat()
    {
        using var mat = CreateMat(100, 80);
        var outer = new MatRoi(mat, new Rectangle(0, 0, 80, 60), leaveOpen: false);
        var inner = new MatRoi(outer, new Rectangle(10, 10, 30, 20));

        inner.Dispose();

        // SourceMat must still be alive — outer still owns it
        Assert.False(mat.IsEmpty);

        outer.Dispose();
    }

    #endregion

    #region IsSourceSameSizeOfRoi

    [Fact]
    public void IsSourceSameSizeOfRoi_RoiMatchesSourceDimensions_ReturnsTrue()
    {
        using var mat = CreateMat(100, 80);

        using var matRoi = new MatRoi(mat, new Rectangle(0, 0, 100, 80));

        Assert.True(matRoi.IsSourceSameSizeOfRoi);
    }

    [Fact]
    public void IsSourceSameSizeOfRoi_RoiSmallerThanSource_ReturnsFalse()
    {
        using var mat = CreateMat(100, 80);

        using var matRoi = new MatRoi(mat, new Rectangle(0, 0, 50, 40));

        Assert.False(matRoi.IsSourceSameSizeOfRoi);
    }

    #endregion

    #region Pixel data correctness

    [Fact]
    public void RoiMat_RoiInBlackRegion_AllPixelsAreZero()
    {
        using var mat = CreateMat(100, 80); // rows 0–39 = black
        var roiRect = new Rectangle(5, 5, 40, 30); // fully within top half

        using var matRoi = new MatRoi(mat, roiRect);

        var bytes = ToContiguousBytes(matRoi.RoiMat);
        Assert.All(bytes, b => Assert.Equal(0, b));
    }

    [Fact]
    public void RoiMat_RoiInWhiteRegion_AllPixelsAre255()
    {
        using var mat = CreateMat(100, 80); // rows 40–79 = white
        var roiRect = new Rectangle(5, 45, 40, 20); // fully within bottom half

        using var matRoi = new MatRoi(mat, roiRect);

        var bytes = ToContiguousBytes(matRoi.RoiMat);
        Assert.All(bytes, b => Assert.Equal(255, b));
    }

    #endregion

    #region Disposal behaviour

    [Fact]
    public void Dispose_LeaveOpenTrue_SourceMatRemainsUsable()
    {
        using var mat = CreateMat();
        var matRoi = new MatRoi(mat, new Rectangle(0, 0, 50, 50), leaveOpen: true);

        matRoi.Dispose();

        Assert.False(mat.IsEmpty);
        Assert.Equal(new Size(100, 80), mat.Size);
    }

    #endregion
}
