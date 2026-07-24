using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;

namespace EmguExtensions.Tests;

/// <summary>
/// Regression tests for general extension method safety and edge cases.
/// </summary>
public class UnitTestExtensionFixes
{
    [Fact]
    public void CreateVector_SourceMutation_AffectMat()
    {
        byte[] source = [1, 2, 3, 4, 5];

        using var mat = EmguCvExtensions.CreateVector(source);
        source.AsSpan().Clear();

        Assert.Equal(new byte[] { 0, 0, 0, 0, 0 }, mat.ToArray());
    }

    [Fact]
    public void GetSpan_OverflowingElementOffset_ThrowsArgumentOutOfRangeException()
    {
        using var mat = new Mat(1, 16, DepthType.Cv8U, 1);

        Assert.Throws<ArgumentOutOfRangeException>(() => mat.GetSpan<int>(0, int.MaxValue));
    }

    [Fact]
    public void GetMemory_OverflowingElementOffset_ThrowsArgumentOutOfRangeException()
    {
        using var mat = new Mat(1, 16, DepthType.Cv8U, 1);

        Assert.Throws<ArgumentOutOfRangeException>(() => mat.GetMemory<int>(0, int.MaxValue));
    }

    [Theory]
    [InlineData(-1, 0)]
    [InlineData(-10, 1)]
    public void GetSpan_NegativeLength_ThrowsArgumentOutOfRangeException(int length, int offset)
    {
        using var mat = new Mat(1, 16, DepthType.Cv8U, 1);

        Assert.Throws<ArgumentOutOfRangeException>(() => mat.GetSpan<byte>(length, offset));
    }

    [Fact]
    public void GetByte_NonContinuousMat_UsesPhysicalRowStride()
    {
        using var source = new Mat(4, 10, DepthType.Cv8U, 1);
        var sourceBytes = source.GetSpan<byte>();
        for (var i = 0; i < sourceBytes.Length; i++)
        {
            sourceBytes[i] = (byte)i;
        }

        using var roi = new Mat(source, new Rectangle(2, 1, 4, 3));

        Assert.False(roi.IsContinuous);
        Assert.Equal(22, roi.GetByte(0, 1));
        Assert.Equal(new byte[] { 12, 13, 14, 15, 22, 23, 24, 25, 32, 33, 34, 35 }, roi.ToArray());
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(16)]
    public void GetByte_OffsetOutsideData_ThrowsArgumentOutOfRangeException(int offset)
    {
        using var mat = new Mat(1, 16, DepthType.Cv8U, 1);

        Assert.Throws<ArgumentOutOfRangeException>(() => mat.GetByte(offset));
    }

    [Fact]
    public void GetSpan2D_OverflowingRoiCoordinates_ThrowsArgumentOutOfRangeException()
    {
        using var mat = new Mat(4, 4, DepthType.Cv8U, 1);
        var roi = new Rectangle(int.MaxValue, 0, 10, 1);

        Assert.Throws<ArgumentOutOfRangeException>(() => mat.GetSpan2D<byte>(roi));
    }

    [Fact]
    public void GetSpan2D_ZeroWidthRoi_ReturnsEmpty()
    {
        using var mat = new Mat(4, 4, DepthType.Cv8U, 1);

        var result = mat.GetSpan2D<byte>(new Rectangle(2, 1, 0, 2));

        Assert.True(result.IsEmpty);
    }

    [Fact]
    public void FindLength_ExtremeCoordinates_DoesNotOverflow()
    {
        var start = new Point(int.MinValue, int.MinValue);
        var end = new Point(int.MaxValue, int.MaxValue);
        var delta = (double)int.MaxValue - int.MinValue;
        var expected = Math.Sqrt(delta * delta + delta * delta);

        var actual = PointExtensions.FindLength(start, end);

        Assert.Equal(expected, actual);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void PolygonCalculations_FewerThanThreeSides_ThrowArgumentOutOfRangeException(int sides)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            DrawingExtensions.CalculatePolygonSideLengthFromRadius(10, sides));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            DrawingExtensions.CalculatePolygonVerticalLengthFromRadius(10, sides));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            DrawingExtensions.CalculatePolygonRadiusFromSideLength(10, sides));
    }
}
