using System.Drawing;
using System.IO.Compression;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace EmguExtensions.Tests;

/// <summary>
/// Tests for extension methods in EmguExtensions.cs:
/// DepthType.ByteCount, IInputArray pixel stats, GetPngBytes, CopyTo, CopyToCenter,
/// GetSpan/GetSpan2D/GetRowSpan, Roi/SafeRoi/RoiFromCenter, pixel accessors,
/// Mat properties, and initializer helpers.
/// </summary>
/// <remarks>
/// Design notes:
/// - <c>Roi(Rectangle)</c> returns a submatrix <em>view</em>; callers must dispose it.
/// - <c>GetSpan2D&lt;T&gt;(Rectangle)</c> pointer arithmetic is tested with pixel-level assertions.
/// - Non-continuous Mat: created via a column-subset ROI (<c>width &lt; src.Width</c>).
/// </remarks>
public class UnitTestEmguExtensions
{
    // 100×80 single-channel mat: top half black (0), bottom half white (255).
    private static Mat CreateGradientMat(int width = 100, int height = 80)
    {
        var mat = EmguExtensions.InitMat(new Size(width, height));
        CvInvoke.Rectangle(mat, new Rectangle(0, height / 2, width, height - height / 2),
            EmguExtensions.WhiteColor, -1);
        return mat;
    }

    // Solid-colour single-channel mat.
    private static Mat CreateSolidMat(int width, int height, byte value = 0)
    {
        var mat = new Mat(height, width, DepthType.Cv8U, 1);
        mat.SetTo(new MCvScalar(value));
        return mat;
    }

    // Returns a non-continuous submatrix view by narrowing the column count.
    private static Mat CreateNonContinuousMat(Mat src)
        => new(src, new Rectangle(0, 0, src.Width / 2, src.Height));

    #region DepthType.ByteCount

    [Theory]
    [InlineData(DepthType.Cv8U,  1)]
    [InlineData(DepthType.Cv8S,  1)]
    [InlineData(DepthType.Cv16U, 2)]
    [InlineData(DepthType.Cv16S, 2)]
    [InlineData(DepthType.Cv32S, 4)]
    [InlineData(DepthType.Cv32F, 4)]
    [InlineData(DepthType.Cv64F, 8)]
    public void ByteCount_KnownDepthType_ReturnsCorrectByteCount(DepthType depth, byte expected)
    {
        Assert.Equal(expected, depth.ByteCount);
    }

    [Fact]
    public void ByteCount_UnknownDepthType_ThrowsArgumentOutOfRangeException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => _ = ((DepthType)99).ByteCount);
    }

    #endregion

    #region IsAllZero / HasNonZero / CountNonZero

    [Fact]
    public void IsAllZero_AllZeroMat_ReturnsTrue()
    {
        using var mat = EmguExtensions.InitMat(new Size(10, 10));
        Assert.True(mat.IsAllZero);
    }

    [Fact]
    public void IsAllZero_MatWithWhitePixel_ReturnsFalse()
    {
        using var mat = CreateGradientMat(10, 10);
        Assert.False(mat.IsAllZero);
    }

    [Fact]
    public void HasNonZero_AllZeroMat_ReturnsFalse()
    {
        using var mat = EmguExtensions.InitMat(new Size(10, 10));
        Assert.False(mat.HasNonZero);
    }

    [Fact]
    public void HasNonZero_MatWithWhitePixel_ReturnsTrue()
    {
        using var mat = CreateGradientMat();
        Assert.True(mat.HasNonZero);
    }

    [Fact]
    public void CountNonZero_AllZeroMat_ReturnsZero()
    {
        using var mat = EmguExtensions.InitMat(new Size(10, 10));
        Assert.Equal(0, mat.CountNonZero);
    }

    [Fact]
    public void CountNonZero_PartiallyWhiteMat_ReturnsCorrectCount()
    {
        // Bottom half (40×100 = 4 000 pixels) is white.
        using var mat = CreateGradientMat(100, 80);
        Assert.Equal(4000, mat.CountNonZero);
    }

    #endregion

    #region GetPngBytes

    [Fact]
    public void GetPngBytes_ValidMat_ReturnsPngBytes()
    {
        using var mat = CreateGradientMat();
        var bytes = mat.GetPngBytes();
        Assert.NotEmpty(bytes);
        // PNG magic bytes: 0x89 0x50 0x4E 0x47
        Assert.Equal(0x89, bytes[0]);
        Assert.Equal(0x50, bytes[1]);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(9)]
    public void GetPngBytes_ValidCompressionLevel_ReturnsPngBytes(int level)
    {
        using var mat = CreateGradientMat();
        var bytes = mat.GetPngBytes(level);
        Assert.NotEmpty(bytes);
        Assert.Equal(0x89, bytes[0]);
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(10)]
    public void GetPngBytes_InvalidCompressionLevel_ThrowsArgumentOutOfRangeException(int level)
    {
        using var mat = CreateGradientMat();
        Assert.Throws<ArgumentOutOfRangeException>(() => mat.GetPngBytes(level));
    }

    [Theory]
    [InlineData(CompressionLevel.NoCompression, 0)]
    [InlineData(CompressionLevel.Fastest,        1)]
    [InlineData(CompressionLevel.Optimal,        3)]
    [InlineData(CompressionLevel.SmallestSize,   9)]
    public void GetPngBytes_CompressionLevelEnum_ProducesValidPng(CompressionLevel level, int _)
    {
        using var mat = CreateGradientMat();
        var bytes = mat.GetPngBytes(level);
        Assert.NotEmpty(bytes);
        Assert.Equal(0x89, bytes[0]);
    }

    [Fact]
    public void GetPngBytes_HigherCompressionProducesSmallerOrEqualOutput()
    {
        using var mat = CreateGradientMat();
        var noComp     = mat.GetPngBytes(0);
        var maxComp    = mat.GetPngBytes(9);
        Assert.True(maxComp.Length <= noComp.Length,
            $"Level-9 ({maxComp.Length}) should be ≤ level-0 ({noComp.Length})");
    }

    #endregion

    #region CopyTo(Mat, Point, Mat?)

    [Fact]
    public void CopyTo_WithZeroOffset_CopiesEntireSource()
    {
        using var src = CreateSolidMat(50, 40, 200);
        using var dst = CreateSolidMat(50, 40, 0);

        src.CopyTo(dst);

        Assert.All(dst.ToArray(), b => Assert.Equal(200, b));
    }

    [Fact]
    public void CopyTo_WithPositiveOffset_OnlyOverlappingRegionCopied()
    {
        using var src = CreateSolidMat(40, 30, 255);
        using var dst = CreateSolidMat(100, 80, 0);

        // Place src at (20, 10) inside dst
        src.CopyTo(dst, new Point(20, 10));

        // Pixel at dst(20, 10) should be white
        Assert.Equal(255, dst.GetByte(20, 10));
        // Pixel at dst(0, 0) should still be black
        Assert.Equal(0, dst.GetByte(0, 0));
    }

    [Fact]
    public void CopyTo_OffsetPlacesSrcCompletelyOutsideDst_NothingCopied()
    {
        using var src = CreateSolidMat(10, 10, 255);
        using var dst = CreateSolidMat(50, 50, 0);

        src.CopyTo(dst, new Point(200, 200)); // completely outside

        Assert.All(dst.ToArray(), b => Assert.Equal(0, b));
    }

    [Fact]
    public void CopyTo_NegativeOffset_CopiesClippedRegion()
    {
        using var src = CreateSolidMat(40, 40, 255);
        using var dst = CreateSolidMat(50, 50, 0);

        // Place src starting at (-10, -10) — only a 30×30 overlap
        src.CopyTo(dst, new Point(-10, -10));

        // Pixel at dst(0, 0) should be white (inside overlap)
        Assert.Equal(255, dst.GetByte(0, 0));
        // Pixel at dst(35, 35) should be black (outside the 30×30 overlap)
        Assert.Equal(0, dst.GetByte(35, 35));
    }

    #endregion

    #region CopyToCenter

    [Fact]
    public void CopyToCenter_SameSize_CopiesAll()
    {
        using var src = CreateSolidMat(50, 40, 200);
        using var dst = CreateSolidMat(50, 40, 0);

        src.CopyToCenter(dst);

        Assert.All(dst.ToArray(), b => Assert.Equal(200, b));
    }

    [Fact]
    public void CopyToCenter_SrcSmallerThanDst_PlacedInCenter()
    {
        using var src = CreateSolidMat(20, 20, 255);
        using var dst = CreateSolidMat(60, 60, 0);

        src.CopyToCenter(dst);

        // Centre of dst should be white: pixel at (30, 30) is inside the 20×20 copy region
        Assert.Equal(255, dst.GetByte(30, 30));
        // Corners of dst should remain black
        Assert.Equal(0, dst.GetByte(0, 0));
        Assert.Equal(0, dst.GetByte(59, 59));
    }

    [Fact]
    public void CopyToCenter_SrcLargerThanDst_CentreOfSrcFillsDst()
    {
        // Uniform 100×100 source — value 77 everywhere. Whatever the crop, dst should be 77.
        using var src = CreateSolidMat(100, 100, 77);
        using var dst = CreateSolidMat(20, 20, 0);

        src.CopyToCenter(dst);

        Assert.All(dst.ToArray(), b => Assert.Equal(77, b));
    }

    [Fact]
    public void CopyToCenter_SrcWiderButShorter_HandledCorrectly()
    {
        // src: 80 wide × 20 tall (white); dst: 40 wide × 60 tall (black)
        using var src = CreateSolidMat(80, 20, 255);
        using var dst = CreateSolidMat(40, 60, 0);

        src.CopyToCenter(dst);

        // Vertical centre of dst (~row 20) should be white
        Assert.Equal(255, dst.GetByte(20, 20));
        // Rows outside the copy height should remain black
        Assert.Equal(0, dst.GetByte(20, 0));
        Assert.Equal(0, dst.GetByte(20, 59));
    }

    #endregion

    #region GetSpan<T>

    [Fact]
    public void GetSpan_DefaultParams_ReturnsFullDataSpan()
    {
        using var mat = CreateSolidMat(10, 10, 42);
        var span = mat.GetSpan<byte>();
        Assert.Equal(100, span.Length);
        Assert.All(span.ToArray(), b => Assert.Equal(42, b));
    }

    [Fact]
    public void GetSpan_WithOffset_SkipsCorrectNumberOfElements()
    {
        using var mat = CreateSolidMat(10, 10, 255);
        var span = mat.GetSpan<byte>(length: 0, offset: 10); // skip first 10 bytes
        Assert.Equal(90, span.Length);
        Assert.All(span.ToArray(), b => Assert.Equal(255, b));
    }

    [Fact]
    public void GetSpan_NegativeOffset_ThrowsArgumentOutOfRangeException()
    {
        using var mat = CreateSolidMat(10, 10);
        Assert.Throws<ArgumentOutOfRangeException>(() => mat.GetSpan<byte>(0, -1));
    }

    [Fact]
    public void GetSpan_LengthExceedsBounds_ThrowsArgumentOutOfRangeException()
    {
        using var mat = CreateSolidMat(10, 10);
        // Named params are required to resolve to our C# 14 extension rather than
        // Emgu.CV's class member GetSpan<T>(int size) (which has a different param name).
        Assert.Throws<ArgumentOutOfRangeException>(() => mat.GetSpan<byte>(length: 200, offset: 0));
    }

    [Fact]
    public void GetSpan_NonContinuousMat_ThrowsNotSupportedException()
    {
        using var src = CreateSolidMat(20, 10, 0);
        using var nonContinuous = CreateNonContinuousMat(src);
        Assert.Throws<NotSupportedException>(() => nonContinuous.GetSpan<byte>());
    }

    #endregion

    #region GetSpan2D<T>

    [Fact]
    public void GetSpan2D_ContinuousMat_DimensionsMatchMat()
    {
        using var mat = CreateSolidMat(30, 20, 128);
        var span = mat.GetSpan2D<byte>();
        Assert.Equal(20, span.Height);
        Assert.Equal(30, span.Width);
    }

    [Fact]
    public void GetSpan2D_ContinuousMat_PixelValuesCorrect()
    {
        using var mat = CreateGradientMat(10, 10); // rows 0-4 black, 5-9 white
        var span = mat.GetSpan2D<byte>();
        Assert.Equal(0, span[0, 0]);   // top-left black
        Assert.Equal(255, span[9, 0]); // bottom-left white
    }

    [Fact]
    public void GetSpan2D_WithRoi_DimensionsMatchRoi()
    {
        using var mat = CreateSolidMat(100, 80);
        var roi = new Rectangle(10, 5, 40, 20);
        var span = mat.GetSpan2D<byte>(roi);
        Assert.Equal(20, span.Height);
        Assert.Equal(40, span.Width);
    }

    [Fact]
    public void GetSpan2D_WithRoi_EmptyRectangle_ReturnsEmptySpan()
    {
        using var mat = CreateSolidMat(100, 80);
        var span = mat.GetSpan2D<byte>(Rectangle.Empty);
        Assert.True(span.IsEmpty);
    }

    [Fact]
    public void GetSpan2D_WithRoi_RoiExceedsRight_ThrowsArgumentOutOfRangeException()
    {
        using var mat = CreateSolidMat(100, 80);
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            mat.GetSpan2D<byte>(new Rectangle(90, 0, 20, 10))); // right = 110 > 100
    }

    [Fact]
    public void GetSpan2D_WithRoi_NegativeX_ThrowsArgumentOutOfRangeException()
    {
        using var mat = CreateSolidMat(100, 80);
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            mat.GetSpan2D<byte>(new Rectangle(-1, 0, 10, 10)));
    }

    [Fact]
    public void GetSpan2D_WithRoi_PixelValuesMatchSource()
    {
        using var mat = CreateGradientMat(100, 80); // rows 0-39 black, 40-79 white
        var whiteRoi = new Rectangle(0, 40, 100, 40);
        var span = mat.GetSpan2D<byte>(whiteRoi);
        for (int r = 0; r < span.Height; r++)
            Assert.All(span.GetRowSpan(r).ToArray(), b => Assert.Equal(255, b));
    }

    #endregion

    #region GetRowSpan<T>

    [Fact]
    public void GetRowSpan_Row0_ReturnsFirstRowData()
    {
        using var mat = CreateGradientMat(10, 10); // row 0 = black
        var span = mat.GetRowSpan<byte>(y: 0);
        Assert.Equal(10, span.Length);
        Assert.All(span.ToArray(), b => Assert.Equal(0, b));
    }

    [Fact]
    public void GetRowSpan_LastRow_ReturnsLastRowData()
    {
        using var mat = CreateGradientMat(10, 10); // row 9 = white
        var span = mat.GetRowSpan<byte>(y: 9);
        Assert.All(span.ToArray(), b => Assert.Equal(255, b));
    }

    [Fact]
    public void GetRowSpan_NegativeOffset_ThrowsArgumentOutOfRangeException()
    {
        using var mat = CreateSolidMat(10, 10);
        Assert.Throws<ArgumentOutOfRangeException>(() => mat.GetRowSpan<byte>(0, 0, -1));
    }

    [Fact]
    public void GetRowSpan_NegativeRow_ThrowsArgumentOutOfRangeException()
    {
        using var mat = CreateSolidMat(10, 10);
        Assert.Throws<ArgumentOutOfRangeException>(() => mat.GetRowSpan<byte>(-1));
    }

    [Fact]
    public void GetRowSpan_RowEqualToHeight_ThrowsArgumentOutOfRangeException()
    {
        using var mat = CreateSolidMat(10, 10);
        Assert.Throws<ArgumentOutOfRangeException>(() => mat.GetRowSpan<byte>(10));
    }

    #endregion

    #region Pixel accessors (GetByte / SetByte)

    [Fact]
    public void GetByte_ByPosition_ReturnsCorrectValue()
    {
        using var mat = CreateSolidMat(10, 10, 77);
        Assert.Equal(77, mat.GetByte(0));
        Assert.Equal(77, mat.GetByte(50));
    }

    [Fact]
    public void GetByte_ByXY_ReturnsCorrectValue()
    {
        using var mat = CreateGradientMat(10, 10); // row 5 onwards = white
        Assert.Equal(0,   mat.GetByte(0, 0)); // black
        Assert.Equal(255, mat.GetByte(0, 5)); // white
    }

    [Fact]
    public void GetByte_ByPoint_ReturnsCorrectValue()
    {
        using var mat = CreateSolidMat(10, 10, 42);
        Assert.Equal(42, mat.GetByte(new Point(3, 3)));
    }

    [Fact]
    public void GetByte_EmptyMat_ThrowsInvalidOperationException()
    {
        using var mat = new Mat();
        Assert.Throws<InvalidOperationException>(() => mat.GetByte(0));
    }

    [Fact]
    public void SetByte_ByXY_WritesCorrectValue()
    {
        using var mat = CreateSolidMat(10, 10, 0);
        mat.SetByte(5, 3, 200);
        Assert.Equal(200, mat.GetByte(5, 3));
    }

    [Fact]
    public void SetByte_ByPosition_WritesCorrectValue()
    {
        using var mat = CreateSolidMat(10, 10, 0);
        mat.SetByte(0, (byte)99);
        Assert.Equal(99, mat.GetByte(0));
    }

    [Fact]
    public void SetByte_ByPoint_WritesCorrectValue()
    {
        using var mat = CreateSolidMat(10, 10, 0);
        mat.SetByte(new Point(7, 2), 123);
        Assert.Equal(123, mat.GetByte(new Point(7, 2)));
    }

    [Fact]
    public void SetByte_Array_WritesAllValues()
    {
        using var mat = CreateSolidMat(10, 10, 0);
        var patch = new byte[] { 1, 2, 3 };
        mat.SetByte(0, patch);
        Assert.Equal(1, mat.GetByte(0));
        Assert.Equal(2, mat.GetByte(1));
        Assert.Equal(3, mat.GetByte(2));
    }

    [Fact]
    public void SetByte_NullArray_ThrowsArgumentNullException()
    {
        using var mat = CreateSolidMat(10, 10, 0);
        Assert.Throws<ArgumentNullException>(() => mat.SetByte(0, (byte[])null!));
    }

    [Fact]
    public void SetByte_EmptyMat_ThrowsInvalidOperationException()
    {
        using var mat = new Mat();
        Assert.Throws<InvalidOperationException>(() => mat.SetByte(0, (byte)1));
    }

    #endregion

    #region Properties (RealStep, LengthInt32, LengthInt64, CenterPoint)

    [Fact]
    public void RealStep_SingleChannelMat_EqualsWidth()
    {
        using var mat = CreateSolidMat(64, 32);
        Assert.Equal(64, mat.RealStep);
    }

    [Fact]
    public void LengthInt32_SingleChannelMat_EqualsWidthTimesHeight()
    {
        using var mat = CreateSolidMat(100, 80);
        Assert.Equal(8000, mat.LengthInt32);
    }

    [Fact]
    public void LengthInt64_SingleChannelMat_EqualsWidthTimesHeight()
    {
        using var mat = CreateSolidMat(100, 80);
        Assert.Equal(8000L, mat.LengthInt64);
    }

    [Fact]
    public void CenterPoint_EvenDimensions_IsWidthOver2AndHeightOver2()
    {
        using var mat = CreateSolidMat(100, 80);
        Assert.Equal(new Point(50, 40), mat.CenterPoint);
    }

    [Fact]
    public void CenterPoint_OddDimensions_IsTruncated()
    {
        using var mat = CreateSolidMat(101, 81);
        Assert.Equal(new Point(50, 40), mat.CenterPoint);
    }

    #endregion

    #region GetByteCount / GetPixelPos

    [Fact]
    public void GetByteCount_SingleChannelMat_EqualsPixelCount()
    {
        using var mat = CreateSolidMat(10, 10);
        Assert.Equal(5, mat.GetByteCount(5));
    }

    [Fact]
    public void GetPixelPos_ByXY_ReturnsCorrectOffset()
    {
        using var mat = CreateSolidMat(100, 80); // RealStep = 100
        Assert.Equal(0,   mat.GetPixelPos(0, 0));
        Assert.Equal(100, mat.GetPixelPos(0, 1));  // y=1 → 1*100
        Assert.Equal(203, mat.GetPixelPos(3, 2));  // y=2, x=3 → 200+3
    }

    #endregion

    #region Initialiser helpers (New, NewZeros, NewSetTo)

    [Fact]
    public void New_ReturnsMatWithSameSizeAndType()
    {
        using var src = new Mat(40, 60, DepthType.Cv16U, 3);
        using var result = src.New();
        Assert.Equal(src.Size, result.Size);
        Assert.Equal(src.Depth, result.Depth);
        Assert.Equal(src.NumberOfChannels, result.NumberOfChannels);
        Assert.NotSame(src, result);
    }

    [Fact]
    public void NewZeros_ReturnsAllZeroMat()
    {
        using var src = CreateSolidMat(20, 15, 200);
        using var result = src.NewZeros();
        Assert.Equal(src.Size, result.Size);
        Assert.All(result.ToArray(), b => Assert.Equal(0, b));
    }

    [Fact]
    public void NewSetTo_ReturnsMatFilledWithColor()
    {
        using var src = CreateSolidMat(10, 10, 0);
        using var result = src.NewSetTo(new MCvScalar(77));
        Assert.All(result.ToArray(), b => Assert.Equal(77, b));
    }

    #endregion

    #region SafeRoi

    [Fact]
    public void SafeRoi_RoiWithinBounds_ReturnsSameRoi()
    {
        using var mat = CreateSolidMat(100, 80);
        var roi = new Rectangle(10, 10, 30, 20);

        using var result = mat.SafeRoi(roi, out var outRoi);

        Assert.Equal(roi, outRoi);
        Assert.Equal(roi.Size, result.Size);
    }

    [Fact]
    public void SafeRoi_RoiExceedsBounds_ClampsToMatEdge()
    {
        using var mat = CreateSolidMat(100, 80);
        // Right=120, Bottom=100 — both exceed bounds
        using var result = mat.SafeRoi(new Rectangle(80, 60, 40, 40), out var outRoi);
        Assert.Equal(new Rectangle(80, 60, 20, 20), outRoi);
    }

    [Fact]
    public void SafeRoi_RoiCompletelyOutside_ReturnsEmptyMatAndEmptyRectangle()
    {
        using var mat = CreateSolidMat(100, 80);
        using var result = mat.SafeRoi(new Rectangle(200, 200, 10, 10), out var outRoi);
        Assert.Equal(Rectangle.Empty, outRoi);
        Assert.True(result.IsEmpty);
    }

    [Fact]
    public void SafeRoi_WithPadding_ExpandsRoi()
    {
        using var mat = CreateSolidMat(100, 80);
        var inner = new Rectangle(20, 20, 30, 20);
        using var result = mat.SafeRoi(inner, out var outRoi, padLeft: 5, padTop: 5, padRight: 5, padBottom: 5);
        Assert.Equal(new Rectangle(15, 15, 40, 30), outRoi);
    }

    #endregion

    #region RoiFromCenter

    [Fact]
    public void RoiFromCenter_SameSize_ReturnsViewWithSameSize()
    {
        using var mat = CreateSolidMat(100, 80);
        using var result = mat.RoiFromCenter(mat.Size);
        Assert.Equal(mat.Size, result.Size);
    }

    [Fact]
    public void RoiFromCenter_SmallerSize_ReturnsCentredCrop()
    {
        using var mat = CreateGradientMat(100, 80); // rows 40-79 white
        // Request 20×20 crop centred at (50, 40) — straddles the black/white boundary
        using var result = mat.RoiFromCenter(new Size(20, 20));
        Assert.Equal(new Size(20, 20), result.Size);
    }

    [Fact]
    public void RoiFromCenter_SizeExceedsMat_ClampedToMatBounds()
    {
        using var mat = CreateSolidMat(30, 20, 128);
        using var result = mat.RoiFromCenter(new Size(200, 200));
        // SafeRoi clamps — result must be non-empty and ≤ mat size
        Assert.False(result.IsEmpty);
        Assert.True(result.Width <= mat.Width);
        Assert.True(result.Height <= mat.Height);
    }

    #endregion

    #region Transform methods

    [Fact]
    public void Rotate_OutputIdentity_CopiesSourceToDestination()
    {
        using var src = CreateSolidMat(10, 10, 123);
        using var dst = new Mat();

        src.RotateFromCenter(dst, angle: 0);

        Assert.Equal(src.Size, dst.Size);
        Assert.Equal(src.Depth, dst.Depth);
        Assert.Equal(src.NumberOfChannels, dst.NumberOfChannels);
        Assert.Equal(src.ToArray(), dst.ToArray());
    }

    [Fact]
    public void ShrinkToFitPreserveAspect_OutputWhenSourceAlreadyFits_CopiesSourceToDestination()
    {
        using var src = CreateSolidMat(10, 10, 123);
        using var dst = new Mat();

        var shrunk = src.ShrinkToFitPreserveAspect(dst, maxWidth: 100, maxHeight: 100);

        Assert.False(shrunk);
        Assert.Equal(src.Size, dst.Size);
        Assert.Equal(src.Depth, dst.Depth);
        Assert.Equal(src.NumberOfChannels, dst.NumberOfChannels);
        Assert.Equal(src.ToArray(), dst.ToArray());
    }

    [Fact]
    public void ShrinkToFitPreserveAspect_InPlaceWhenSourceExceedsBounds_ReturnsTrueAndShrinks()
    {
        using var src = CreateSolidMat(100, 50, 123);

        var shrunk = src.ShrinkToFitPreserveAspect(maxWidth: 50, maxHeight: 50);

        Assert.True(shrunk);
        Assert.Equal(new Size(50, 25), src.Size);
    }

    [Fact]
    public void ShrinkToFitPreserveAspect_OutputWhenSourceExceedsBounds_ReturnsTrueAndShrinks()
    {
        using var src = CreateSolidMat(100, 50, 123);
        using var dst = new Mat();

        var shrunk = src.ShrinkToFitPreserveAspect(dst, maxWidth: 50, maxHeight: 50);

        Assert.True(shrunk);
        Assert.Equal(new Size(50, 25), dst.Size);
        Assert.Equal(src.Depth, dst.Depth);
        Assert.Equal(src.NumberOfChannels, dst.NumberOfChannels);
    }

    #endregion

    #region FillSpan<T>

    [Fact]
    public void FillSpan_ValueAboveThreshold_FillsRegion()
    {
        using var mat = CreateSolidMat(10, 10, 0);
        int pos = 0;
        mat.FillSpan<byte>(ref pos, length: 5, value: 200, valueMinThreshold: 1);

        Assert.Equal(5, pos);
        Assert.Equal(200, mat.GetByte(0));
        Assert.Equal(200, mat.GetByte(4));
        Assert.Equal(0,   mat.GetByte(5)); // outside the fill
    }

    [Fact]
    public void FillSpan_ValueBelowThreshold_SkipsFillButAdvancesPosition()
    {
        using var mat = CreateSolidMat(10, 10, 0);
        int pos = 0;
        mat.FillSpan<byte>(ref pos, length: 5, value: 0, valueMinThreshold: 1);

        Assert.Equal(5, pos);          // position advanced
        Assert.Equal(0, mat.GetByte(0)); // data unchanged
    }

    [Fact]
    public void FillSpan_ZeroLength_NoChangeAndPositionUnchanged()
    {
        using var mat = CreateSolidMat(10, 10, 50);
        int pos = 3;
        mat.FillSpan<byte>(ref pos, length: 0, value: 255, valueMinThreshold: 0);

        Assert.Equal(3, pos); // unchanged
    }

    #endregion
}
