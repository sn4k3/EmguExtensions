using Emgu.CV;
using Emgu.CV.CvEnum;

namespace EmguExtensions.Tests;

/// <summary>
/// Tests for FindFirst* extension methods on <see cref="Mat"/>.
/// </summary>
/// <remarks>
/// Note: <c>length</c> in these APIs is passed as the first argument to
/// <c>GetReadOnlySpan&lt;T&gt;(length, offset)</c>, so it controls how many elements
/// are scanned from <c>startPos</c>. Passing <c>length=0</c> scans to the end.
/// </remarks>
public class UnitTestFindMethods
{
    private static Mat CreateMat(params byte[] pixels)
    {
        var mat = new Mat(1, pixels.Length, DepthType.Cv8U, 1);
        var span = mat.GetSpan<byte>(pixels.Length, 0);
        pixels.CopyTo(span);
        return mat;
    }

    private static Mat CreateEmptyMat() => new();

    #region FindFirstNegativePixel

    [Fact]
    public void FindFirstNegativePixel_FirstPixelIsZero_ReturnsZero()
    {
        using var mat = CreateMat(0, 128, 255);
        Assert.Equal(0, mat.FindFirstNegativePixel<byte>());
    }

    [Fact]
    public void FindFirstNegativePixel_ZeroAfterNonZero_ReturnsCorrectIndex()
    {
        using var mat = CreateMat(128, 255, 0, 200);
        Assert.Equal(2, mat.FindFirstNegativePixel<byte>());
    }

    [Fact]
    public void FindFirstNegativePixel_AllNonZero_ReturnsMinusOne()
    {
        using var mat = CreateMat(1, 128, 255);
        Assert.Equal(-1, mat.FindFirstNegativePixel<byte>());
    }

    [Fact]
    public void FindFirstNegativePixel_WithStartPosBeyondZeroPixel_ReturnsMinusOne()
    {
        using var mat = CreateMat(0, 128, 255);
        Assert.Equal(-1, mat.FindFirstNegativePixel<byte>(startPos: 1));
    }

    #endregion

    #region FindFirstPositivePixel

    [Fact]
    public void FindFirstPositivePixel_FirstPixelNonZero_ReturnsZero()
    {
        using var mat = CreateMat(200, 0, 0);
        Assert.Equal(0, mat.FindFirstPositivePixel<byte>());
    }

    [Fact]
    public void FindFirstPositivePixel_NonZeroAfterZeros_ReturnsCorrectIndex()
    {
        using var mat = CreateMat(0, 0, 128);
        Assert.Equal(2, mat.FindFirstPositivePixel<byte>());
    }

    [Fact]
    public void FindFirstPositivePixel_AllZero_ReturnsMinusOne()
    {
        using var mat = CreateMat(0, 0, 0);
        Assert.Equal(-1, mat.FindFirstPositivePixel<byte>());
    }

    #endregion

    #region FindFirstPixelEqualTo

    [Fact]
    public void FindFirstPixelEqualTo_ValueAtStart_ReturnsZero()
    {
        using var mat = CreateMat(42, 0, 0);
        Assert.Equal(0, mat.FindFirstPixelEqualTo<byte>(42));
    }

    [Fact]
    public void FindFirstPixelEqualTo_ValueAtEnd_ReturnsLastIndex()
    {
        using var mat = CreateMat(0, 0, 99);
        Assert.Equal(2, mat.FindFirstPixelEqualTo<byte>(99));
    }

    [Fact]
    public void FindFirstPixelEqualTo_ValueNotPresent_ReturnsMinusOne()
    {
        using var mat = CreateMat(1, 2, 3);
        Assert.Equal(-1, mat.FindFirstPixelEqualTo<byte>(200));
    }

    [Fact]
    public void FindFirstPixelEqualTo_WithStartPos_SkipsPrecedingPixels()
    {
        using var mat = CreateMat(42, 0, 42);
        Assert.Equal(2, mat.FindFirstPixelEqualTo<byte>(42, startPos: 1));
    }

    [Fact]
    public void FindFirstPixelEqualTo_WithLength_LimitsSearch()
    {
        using var mat = CreateMat(0, 0, 99);
        Assert.Equal(-1, mat.FindFirstPixelEqualTo<byte>(99, length: 2));
    }

    [Fact]
    public void FindFirstPixelEqualTo_SinglePixelMat_MatchFound()
    {
        using var mat = CreateMat(77);
        Assert.Equal(0, mat.FindFirstPixelEqualTo<byte>(77));
    }

    [Fact]
    public void FindFirstPixelEqualTo_SinglePixelMat_NoMatch()
    {
        using var mat = CreateMat(77);
        Assert.Equal(-1, mat.FindFirstPixelEqualTo<byte>(78));
    }

    #endregion

    #region FindFirstPixelLessThan

    [Fact]
    public void FindFirstPixelLessThan_ValueIsMinValue_ReturnsMinusOne()
    {
        using var mat = CreateMat(0, 100);
        Assert.Equal(-1, mat.FindFirstPixelLessThan<byte>(byte.MinValue));
    }

    [Fact]
    public void FindFirstPixelLessThan_PixelLessThanValue_ReturnsIndex()
    {
        using var mat = CreateMat(200, 50, 200);
        Assert.Equal(1, mat.FindFirstPixelLessThan<byte>(100));
    }

    [Fact]
    public void FindFirstPixelLessThan_AllPixelsEqualOrGreater_ReturnsMinusOne()
    {
        using var mat = CreateMat(100, 150, 200);
        Assert.Equal(-1, mat.FindFirstPixelLessThan<byte>(100));
    }

    [Fact]
    public void FindFirstPixelLessThan_FirstPixelMatches_ReturnsZero()
    {
        using var mat = CreateMat(10, 200, 200);
        Assert.Equal(0, mat.FindFirstPixelLessThan<byte>(50));
    }

    [Fact]
    public void FindFirstPixelLessThan_WithStartPos_SkipsEarlierMatch()
    {
        using var mat = CreateMat(10, 200, 10);
        Assert.Equal(2, mat.FindFirstPixelLessThan<byte>(50, startPos: 1));
    }

    #endregion

    #region FindFirstPixelEqualOrLessThan

    [Fact]
    public void FindFirstPixelEqualOrLessThan_ExactMatch_ReturnsIndex()
    {
        using var mat = CreateMat(200, 100, 50);
        Assert.Equal(1, mat.FindFirstPixelEqualOrLessThan<byte>(100));
    }

    [Fact]
    public void FindFirstPixelEqualOrLessThan_MinValue_FindsZero()
    {
        using var mat = CreateMat(128, 0, 200);
        Assert.Equal(1, mat.FindFirstPixelEqualOrLessThan<byte>(0));
    }

    [Fact]
    public void FindFirstPixelEqualOrLessThan_AllAboveValue_ReturnsMinusOne()
    {
        using var mat = CreateMat(200, 200, 200);
        Assert.Equal(-1, mat.FindFirstPixelEqualOrLessThan<byte>(100));
    }

    #endregion

    #region FindFirstPixelGreaterThan

    [Fact]
    public void FindFirstPixelGreaterThan_ValueIsMaxValue_ReturnsMinusOne()
    {
        using var mat = CreateMat(255, 100);
        Assert.Equal(-1, mat.FindFirstPixelGreaterThan<byte>(byte.MaxValue));
    }

    [Fact]
    public void FindFirstPixelGreaterThan_PixelGreaterThanValue_ReturnsIndex()
    {
        using var mat = CreateMat(10, 10, 200);
        Assert.Equal(2, mat.FindFirstPixelGreaterThan<byte>(100));
    }

    [Fact]
    public void FindFirstPixelGreaterThan_AllPixelsEqualOrLess_ReturnsMinusOne()
    {
        using var mat = CreateMat(10, 50, 100);
        Assert.Equal(-1, mat.FindFirstPixelGreaterThan<byte>(100));
    }

    [Fact]
    public void FindFirstPixelGreaterThan_FirstPixelMatches_ReturnsZero()
    {
        using var mat = CreateMat(200, 10, 10);
        Assert.Equal(0, mat.FindFirstPixelGreaterThan<byte>(100));
    }

    #endregion

    #region FindFirstPixelEqualOrGreaterThan

    [Fact]
    public void FindFirstPixelEqualOrGreaterThan_ExactMatch_ReturnsIndex()
    {
        using var mat = CreateMat(10, 100, 200);
        Assert.Equal(1, mat.FindFirstPixelEqualOrGreaterThan<byte>(100));
    }

    [Fact]
    public void FindFirstPixelEqualOrGreaterThan_MaxValue_FindsMaxValue()
    {
        using var mat = CreateMat(100, 200, 255);
        Assert.Equal(2, mat.FindFirstPixelEqualOrGreaterThan<byte>(255));
    }

    [Fact]
    public void FindFirstPixelEqualOrGreaterThan_AllBelowValue_ReturnsMinusOne()
    {
        using var mat = CreateMat(10, 20, 30);
        Assert.Equal(-1, mat.FindFirstPixelEqualOrGreaterThan<byte>(100));
    }

    [Fact]
    public void FindFirstPixelEqualOrGreaterThan_MinValue_ReturnsZeroIfAnyPixelExists()
    {
        using var mat = CreateMat(0, 0, 50);
        Assert.Equal(0, mat.FindFirstPixelEqualOrGreaterThan<byte>(0));
    }

    #endregion
}
