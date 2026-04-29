using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;

namespace EmguExtensions.Tests;

/// <summary>
/// Tests for <see cref="EmguExtensions.ScanStrides"/> and <see cref="EmguExtensions.ScanLines"/>.
/// </summary>
/// <remarks>
/// Design note: <c>GreyLine</c> is a mutable record struct. Record equality compares all properties, so
/// <c>Assert.Equal</c> on two <c>GreyLine</c> instances works as expected without extra helpers.
/// </remarks>
public class UnitTestScanMethods
{
    // Creates a single-row, n-column single-channel mat from the given byte values.
    private static Mat CreateMat1D(params byte[] pixels)
    {
        var mat = new Mat(1, pixels.Length, DepthType.Cv8U, 1);
        var span = mat.GetSpan<byte>(pixels.Length, 0);
        pixels.CopyTo(span);
        return mat;
    }

    // Creates a multi-row mat. Each inner array is one row; all rows must be the same length.
    private static Mat CreateMat2D(byte[][] rows)
    {
        int height = rows.Length;
        int width = rows[0].Length;
        var mat = new Mat(height, width, DepthType.Cv8U, 1);
        for (int y = 0; y < height; y++)
        {
            var span = mat.GetRowSpan<byte>(y);
            rows[y].CopyTo(span);
        }
        return mat;
    }

    private static GreyStride S(int index, int x, int y, uint stride, byte grey)
        => new(index, new Point(x, y), stride, grey);

    private static GreyLine L(int sx, int sy, int ex, int ey, byte grey)
        => new() { StartX = sx, StartY = sy, EndX = ex, EndY = ey, Grey = grey };

    #region ScanStrides — validation

    [Fact]
    public void ScanStrides_NegativeStrideLimit_ThrowsArgumentOutOfRangeException()
    {
        using var mat = CreateMat1D(0, 255);
        Assert.Throws<ArgumentOutOfRangeException>(() => mat.ScanStrides(strideLimit: -1));
    }

    [Fact]
    public void ScanStrides_StrideLimitOne_ThrowsArgumentOutOfRangeException()
    {
        using var mat = CreateMat1D(0, 255);
        Assert.Throws<ArgumentOutOfRangeException>(() => mat.ScanStrides(strideLimit: 1));
    }

    [Fact]
    public void ScanStrides_MultiChannel_ThrowsArgumentOutOfRangeException()
    {
        using var mat = new Mat(1, 4, DepthType.Cv8U, 3);
        Assert.Throws<ArgumentOutOfRangeException>(() => mat.ScanStrides());
    }

    [Fact]
    public void ScanStrides_NullGreyFunc_ThrowsArgumentNullException()
    {
        using var mat = CreateMat1D(0, 255);
        Assert.Throws<ArgumentNullException>(() => mat.ScanStrides(null!));
    }

    #endregion

    #region ScanStrides — empty / trivial

    [Fact]
    public void ScanStrides_EmptyMat_ReturnsEmptyArray()
    {
        using var mat = new Mat();
        var result = mat.ScanStrides();
        Assert.Empty(result);
    }

    [Fact]
    public void ScanStrides_SinglePixel_ReturnsSingleStride()
    {
        using var mat = CreateMat1D(128);
        var result = mat.ScanStrides();
        Assert.Single(result);
        Assert.Equal(S(0, 0, 0, 1, 128), result[0]);
    }

    #endregion

    #region ScanStrides — basic stride splitting

    [Fact]
    public void ScanStrides_TwoDistinctValues_ReturnsTwoStrides()
    {
        // [0, 0, 255, 255] → two strides
        using var mat = CreateMat1D(0, 0, 255, 255);
        var result = mat.ScanStrides();

        Assert.Equal(2, result.Length);
        Assert.Equal(S(0, 0, 0, 2, 0), result[0]);
        Assert.Equal(S(2, 2, 0, 2, 255), result[1]);
    }

    [Fact]
    public void ScanStrides_AlternatingValues_EachPixelOwnStride()
    {
        // [0, 255, 0] → three strides
        using var mat = CreateMat1D(0, 255, 0);
        var result = mat.ScanStrides();

        Assert.Equal(3, result.Length);
        Assert.Equal(S(0, 0, 0, 1, 0), result[0]);
        Assert.Equal(S(1, 1, 0, 1, 255), result[1]);
        Assert.Equal(S(2, 2, 0, 1, 0), result[2]);
    }

    [Fact]
    public void ScanStrides_AllSameValue_ReturnsSingleStride()
    {
        using var mat = CreateMat1D(128, 128, 128, 128);
        var result = mat.ScanStrides();

        Assert.Single(result);
        Assert.Equal(S(0, 0, 0, 4, 128), result[0]);
    }

    #endregion

    #region ScanStrides — strideLimit

    [Fact]
    public void ScanStrides_StrideLimitSplitsRun_ProducesExpectedChunks()
    {
        // [0, 0, 0, 0] with strideLimit=2 → two strides of 2
        using var mat = CreateMat1D(0, 0, 0, 0);
        var result = mat.ScanStrides(strideLimit: 2);

        Assert.Equal(2, result.Length);
        Assert.Equal(S(0, 0, 0, 2, 0), result[0]);
        Assert.Equal(S(2, 2, 0, 2, 0), result[1]);
    }

    #endregion

    #region ScanStrides — excludeBlacks

    [Fact]
    public void ScanStrides_ExcludeBlacks_OmitsZeroStrides()
    {
        // [0, 0, 128, 128, 0] → only the 128 stride returned
        using var mat = CreateMat1D(0, 0, 128, 128, 0);
        var result = mat.ScanStrides(excludeBlacks: true);

        Assert.Single(result);
        Assert.Equal(128, result[0].Grey);
        Assert.Equal(2u, result[0].Stride);
    }

    [Fact]
    public void ScanStrides_ExcludeBlacks_AllBlack_ReturnsEmpty()
    {
        using var mat = CreateMat1D(0, 0, 0);
        var result = mat.ScanStrides(excludeBlacks: true);
        Assert.Empty(result);
    }

    #endregion

    #region ScanStrides — startOnFirstPositivePixel

    [Fact]
    public void ScanStrides_StartOnFirstPositivePixel_SkipsLeadingBlacks()
    {
        // [0, 0, 100, 100] → only the 100 stride
        using var mat = CreateMat1D(0, 0, 100, 100);
        var result = mat.ScanStrides(startOnFirstPositivePixel: true, excludeBlacks: true);

        Assert.Single(result);
        Assert.Equal(100, result[0].Grey);
        Assert.Equal(2u, result[0].Stride);
    }

    [Fact]
    public void ScanStrides_StartOnFirstPositivePixel_AllBlack_ReturnsEmpty()
    {
        using var mat = CreateMat1D(0, 0, 0);
        var result = mat.ScanStrides(startOnFirstPositivePixel: true, excludeBlacks: true);
        Assert.Empty(result);
    }

    #endregion

    #region ScanStrides — breakOnRows

    [Fact]
    public void ScanStrides_BreakOnRows_SameGreyAcrossRows_ProducesOneStridePerRow()
    {
        // 2×2 mat, all 255 → without breakOnRows: 1 stride; with: 2 strides (one per row)
        using var mat = CreateMat2D([
            [255, 255],
            [255, 255]
        ]);

        var withBreak = mat.ScanStrides(breakOnRows: true);
        var withoutBreak = mat.ScanStrides(breakOnRows: false);

        Assert.Equal(2, withBreak.Length);
        Assert.Single(withoutBreak);
    }

    [Fact]
    public void ScanStrides_BreakOnRows_LocationReflectsNewRow()
    {
        using var mat = CreateMat2D([
            [100, 100],
            [100, 100]
        ]);

        var result = mat.ScanStrides(breakOnRows: true);

        Assert.Equal(0, result[0].Location.Y);
        Assert.Equal(1, result[1].Location.Y);
    }

    #endregion

    #region ScanStrides — threshold

    [Fact]
    public void ScanStrides_Threshold_GroupsPixelsBelowAndAbove()
    {
        // [10, 20, 200, 210] with thresholdGrey=128 → [0,0,255,255] after threshold → 2 strides
        using var mat = CreateMat1D(10, 20, 200, 210);
        var result = mat.ScanStrides(thresholdGrey: 128);

        Assert.Equal(2, result.Length);
        Assert.Equal(0, result[0].Grey);
        Assert.Equal(255, result[1].Grey);
    }

    [Fact]
    public void ScanStrides_ThresholdZero_NoThresholdingApplied()
    {
        // thresholdGrey=0 → condition (> MinValue && < MaxValue) is false → raw values used
        using var mat = CreateMat1D(10, 10, 200);
        var result = mat.ScanStrides(thresholdGrey: 0);

        Assert.Equal(2, result.Length);
        Assert.Equal(10, result[0].Grey);
        Assert.Equal(200, result[1].Grey);
    }

    #endregion

    #region ScanStrides — greyFunc overload

    [Fact]
    public void ScanStrides_GreyFunc_TransformsPixelsBeforeGrouping()
    {
        // Raw: [10, 20, 200, 210]. Func inverts: x > 128 → 0, else 255. So [255, 255, 0, 0] → 2 strides.
        using var mat = CreateMat1D(10, 20, 200, 210);
        var result = mat.ScanStrides(x => x > 128 ? (byte)0 : (byte)255);

        Assert.Equal(2, result.Length);
        Assert.Equal(255, result[0].Grey);
        Assert.Equal(0, result[1].Grey);
    }

    [Fact]
    public void ScanStrides_GreyFunc_ExcludeBlacks_OmitsZeroAfterTransform()
    {
        using var mat = CreateMat1D(10, 20, 200, 210);
        var result = mat.ScanStrides(x => x > 128 ? (byte)0 : (byte)255, excludeBlacks: true);

        Assert.Single(result);
        Assert.Equal(255, result[0].Grey);
    }

    [Fact]
    public void ScanStrides_GreyFunc_StrideLimitSplitsRun_ProducesExpectedChunks()
    {
        using var mat = CreateMat1D(1, 2, 3, 4);
        var result = mat.ScanStrides(_ => (byte)255, strideLimit: 2);

        Assert.Equal(2, result.Length);
        Assert.Equal(S(0, 0, 0, 2, 255), result[0]);
        Assert.Equal(S(2, 2, 0, 2, 255), result[1]);
    }

    #endregion

    #region ScanLines — validation

    [Fact]
    public void ScanLines_MultiChannel_ThrowsArgumentOutOfRangeException()
    {
        using var mat = new Mat(4, 4, DepthType.Cv8U, 3);
        Assert.Throws<ArgumentOutOfRangeException>(() => mat.ScanLines());
    }

    [Fact]
    public void ScanLines_NullGreyFunc_ThrowsArgumentNullException()
    {
        using var mat = CreateMat1D(0, 255);
        Assert.Throws<ArgumentNullException>(() => mat.ScanLines(null!));
    }

    #endregion

    #region ScanLines — empty / trivial

    [Fact]
    public void ScanLines_EmptyMat_ReturnsEmptyArray()
    {
        using var mat = new Mat();
        Assert.Empty(mat.ScanLines());
    }

    [Fact]
    public void ScanLines_AllBlack_ReturnsEmptyArray()
    {
        using var mat = CreateMat2D([[0, 0, 0], [0, 0, 0]]);
        Assert.Empty(mat.ScanLines());
    }

    #endregion

    #region ScanLines — horizontal

    [Fact]
    public void ScanLines_SingleWhiteRow_ReturnsOneLine()
    {
        // Row 0: [255, 255, 255] → one line spanning full width
        using var mat = CreateMat2D([[255, 255, 255]]);
        var result = mat.ScanLines();

        Assert.Single(result);
        Assert.Equal(L(0, 0, 2, 0, 255), result[0]);
    }

    [Fact]
    public void ScanLines_MixedRow_TwoLines()
    {
        // Row 0: [255, 0, 128] → two lines: (0,0,0,0,255) and (2,0,2,0,128)
        using var mat = CreateMat2D([[255, 0, 128]]);
        var result = mat.ScanLines();

        Assert.Equal(2, result.Length);
        Assert.Equal(L(0, 0, 0, 0, 255), result[0]);
        Assert.Equal(L(2, 0, 2, 0, 128), result[1]);
    }

    [Fact]
    public void ScanLines_LineInMiddleRow_CorrectY()
    {
        using var mat = CreateMat2D([
            [0, 0, 0],
            [255, 255, 255],
            [0, 0, 0]
        ]);
        var result = mat.ScanLines();

        Assert.Single(result);
        Assert.Equal(1, result[0].StartY);
        Assert.Equal(1, result[0].EndY);
    }

    [Fact]
    public void ScanLines_MultipleRows_ReturnsLinesPerRow()
    {
        using var mat = CreateMat2D([
            [255, 255],
            [0, 0],
            [128, 128]
        ]);
        var result = mat.ScanLines();

        Assert.Equal(2, result.Length);
        Assert.Equal(0, result[0].StartY);
        Assert.Equal(2, result[1].StartY);
    }

    [Fact]
    public void ScanLines_PartialLine_StartsAndEndsAtCorrectX()
    {
        // Row 0: [0, 128, 128, 0] → line from x=1 to x=2
        using var mat = CreateMat2D([[0, 128, 128, 0]]);
        var result = mat.ScanLines();

        Assert.Single(result);
        Assert.Equal(L(1, 0, 2, 0, 128), result[0]);
    }

    [Fact]
    public void ScanLines_WithOffset_ShiftsCoordinates()
    {
        using var mat = CreateMat2D([[255, 255]]);
        var result = mat.ScanLines(offset: new Point(10, 5));

        Assert.Single(result);
        Assert.Equal(10, result[0].StartX);
        Assert.Equal(5, result[0].StartY);
        Assert.Equal(11, result[0].EndX);
        Assert.Equal(5, result[0].EndY);
    }

    #endregion

    #region ScanLines — threshold

    [Fact]
    public void ScanLines_Threshold_GroupsLowAsBlackAndHighAsWhite()
    {
        // Row: [10, 200] with thresholdGrey=128 → [0, 255] → one line for the 255 pixel
        using var mat = CreateMat2D([[10, 200]]);
        var result = mat.ScanLines(thresholdGrey: 128);

        Assert.Single(result);
        Assert.Equal(255, result[0].Grey);
        Assert.Equal(1, result[0].StartX);
    }

    [Fact]
    public void ScanLines_ThresholdZero_NoThresholdingApplied()
    {
        using var mat = CreateMat2D([[10, 200]]);
        var result = mat.ScanLines(thresholdGrey: 0);

        Assert.Equal(2, result.Length);
        Assert.Equal(10, result[0].Grey);
        Assert.Equal(200, result[1].Grey);
    }

    #endregion

    #region ScanLines — greyFunc overload

    [Fact]
    public void ScanLines_GreyFunc_TransformsBeforeGrouping()
    {
        // Row: [10, 200]. Func: x > 128 → 255, else 0. → [0, 255] → 1 line
        using var mat = CreateMat2D([[10, 200]]);
        var result = mat.ScanLines(x => x > 128 ? (byte)255 : (byte)0);

        Assert.Single(result);
        Assert.Equal(255, result[0].Grey);
    }

    [Fact]
    public void ScanLines_GreyFunc_AllMapToZero_ReturnsEmpty()
    {
        using var mat = CreateMat2D([[100, 200]]);
        var result = mat.ScanLines(_ => 0);
        Assert.Empty(result);
    }

    #endregion

    #region ScanLines — vertical

    [Fact]
    public void ScanLines_Vertical_SingleWhiteColumn_ReturnsOneVerticalLine()
    {
        // 3×1 mat (3 rows, 1 col), all 255 → one vertical line at x=0 from y=0 to y=2
        using var mat = CreateMat2D([[255], [255], [255]]);
        var result = mat.ScanLines(vertically: true);

        Assert.Single(result);
        Assert.Equal(L(0, 0, 0, 2, 255), result[0]);
    }

    [Fact]
    public void ScanLines_Vertical_MixedColumn_TwoLines()
    {
        // Col 0: [255, 0, 128] → two lines
        using var mat = CreateMat2D([[255], [0], [128]]);
        var result = mat.ScanLines(vertically: true);

        Assert.Equal(2, result.Length);
        Assert.Equal(L(0, 0, 0, 0, 255), result[0]);
        Assert.Equal(L(0, 2, 0, 2, 128), result[1]);
    }

    [Fact]
    public void ScanLines_Vertical_AllBlack_ReturnsEmpty()
    {
        using var mat = CreateMat2D([[0], [0], [0]]);
        Assert.Empty(mat.ScanLines(vertically: true));
    }

    #endregion
}
