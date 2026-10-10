using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace EmguExtensions.Tests;

public class UnitTestPerspectiveAndMasks
{
    [Fact]
    public void CornerOrdering_AllPermutations_ProduceSameClockwiseOrder()
    {
        PointF[] expected = [new(2, 1), new(8, 2), new(9, 7), new(1, 8)];
        for (var a = 0; a < 4; a++)
        for (var b = 0; b < 4; b++)
        for (var c = 0; c < 4; c++)
        for (var d = 0; d < 4; d++)
        {
            if (new[] { a, b, c, d }.Distinct().Count() != 4) continue;
            PointF[] unordered = [expected[a], expected[b], expected[c], expected[d]];
            var before = unordered.ToArray();
            Assert.Equal(expected, EmguCvExtensions.OrderPerspectiveCorners(unordered));
            Assert.Equal(before, unordered);
        }
    }

    [Fact]
    public void CornerOrdering_DiamondTies_StartAtTop()
    {
        PointF[] unordered = [new(5, 10), new(0, 5), new(10, 5), new(5, 0)];
        PointF[] expected = [new(5, 0), new(10, 5), new(5, 10), new(0, 5)];
        Assert.Equal(expected, EmguCvExtensions.OrderPerspectiveCorners(unordered));
    }

    [Fact]
    public void CornerOrdering_InteriorAndRepeatedPoints_AreRejected()
    {
        Assert.Throws<ArgumentException>(() => EmguCvExtensions.OrderPerspectiveCorners(
            new PointF[] { new(0, 0), new(10, 0), new(0, 10), new(2, 2) }));
        Assert.Throws<ArgumentException>(() => EmguCvExtensions.OrderPerspectiveCorners(
            new PointF[] { new(0, 0), new(10, 0), new(10, 0), new(0, 10) }));
    }

    [Fact]
    public void PerspectiveCrop_Destination_ReusesMemoryAndMatchesAllocatedCrop()
    {
        using var source = EmguCvExtensions.InitMat(new Size(12, 10));
        source.SetTo(new MCvScalar(123));
        PointF[] corners = [new(2, 2), new(8, 2), new(8, 7), new(2, 7)];
        var size = EmguCvExtensions.GetPerspectiveCropSize(corners);
        using var destination = new Mat(size, source.Depth, source.NumberOfChannels);
        var pointer = destination.DataPointer;
        source.PerspectiveCrop(destination, corners, interpolation: Inter.Nearest);
        using var expected = source.CreatePerspectiveCrop(corners, interpolation: Inter.Nearest);
        Assert.Equal(pointer, destination.DataPointer);
        Assert.Equal(expected.ToArray(), destination.ToArray());
    }

    [Fact]
    public void PerspectiveCrop_OverlappingRoi_MatchesIndependentResult()
    {
        using var source = EmguCvExtensions.InitMat(new Size(12, 10));
        for (var y = 0; y < source.Height; y++)
            for (var x = 0; x < source.Width; x++) source.GetSpan2D<byte>()[y, x] = (byte)(y * 12 + x);
        PointF[] corners = [new(2, 2), new(8, 2), new(8, 7), new(2, 7)];
        using var expected = source.CreatePerspectiveCrop(corners, interpolation: Inter.Nearest);
        using var destination = new Mat(source, new Rectangle(1, 1, 7, 6));
        source.PerspectiveCrop(destination, corners, interpolation: Inter.Nearest);
        Assert.Equal(expected.ToArray(), destination.ToArray());
        Assert.Equal(expected.GetReadOnlySpan2D<byte>()[0, 0], source.GetReadOnlySpan2D<byte>()[1, 1]);
    }

    [Fact]
    public void PerspectiveCrop_InPlace_MatchesIndependentResult()
    {
        using var source = EmguCvExtensions.InitMat(new Size(12, 10));
        source.SetTo(new MCvScalar(123));
        PointF[] corners = [new(2, 2), new(8, 2), new(8, 7), new(2, 7)];
        using var expected = source.CreatePerspectiveCrop(corners);
        source.PerspectiveCrop(source, corners);
        Assert.Equal(expected.Size, source.Size);
        Assert.Equal(expected.ToArray(), source.ToArray());
    }

    [Fact]
    public void PerspectiveMapping_ForwardCornersAndInverseInterior_RoundTrip()
    {
        PointF[] corners = [new(2, 1), new(8, 2), new(9, 7), new(1, 8)];
        var size = new Size(30, 20);
        using var forward = EmguCvExtensions.CreatePerspectiveTransform(corners, size);
        using var inverse = EmguCvExtensions.CreatePerspectiveTransform(corners, size, inverse: true);
        PointF[] expected = [new(0, 0), new(29, 0), new(29, 19), new(0, 19)];
        var mapped = forward.MapPerspectivePoints(corners);
        for (var i = 0; i < 4; i++) AssertPointClose(expected[i], mapped[i]);
        PointF[] interior = [new(4, 4), new(6, 5)];
        var roundTrip = inverse.MapPerspectivePoints(forward.MapPerspectivePoints(interior));
        for (var i = 0; i < interior.Length; i++) AssertPointClose(interior[i], roundTrip[i]);
        Assert.Empty(forward.MapPerspectivePoints([]));
    }

    [Fact]
    public void PerspectiveMapping_PointAtInfinity_IsRejected()
    {
        using var transform = Mat.Zeros(3, 3, DepthType.Cv64F, 1);
        var matrix = transform.GetSpan2D<double>();
        matrix[0, 0] = 1; matrix[1, 1] = 1; matrix[2, 0] = 1;
        Assert.Throws<ArgumentOutOfRangeException>(() => transform.MapPerspectivePoints(new PointF[] { new(0, 1) }));
        using var invalid = new Mat(2, 3, DepthType.Cv64F, 1);
        Assert.Throws<ArgumentException>(() => invalid.MapPerspectivePoints([]));
    }

    [Fact]
    public void RotatedRectangleCrop_AxisAligned_MatchesOrderedCorners()
    {
        using var source = EmguCvExtensions.InitMat(new Size(30, 30));
        for (var y = 0; y < source.Height; y++)
            for (var x = 0; x < source.Width; x++) source.GetSpan2D<byte>()[y, x] = (byte)(x + y);
        var rectangle = new RotatedRect(new PointF(15, 15), new SizeF(12, 8), 0);
        using var crop = source.CreatePerspectiveCrop(rectangle, interpolation: Inter.Nearest);
        using var expected = new Mat(source, new Rectangle(9, 11, 13, 9));
        Assert.Equal(expected.Size, crop.Size);
        Assert.Equal(expected.ToArray(), crop.ToArray());
        Assert.True(crop.CountNonZero > 0);
        Assert.Throws<ArgumentOutOfRangeException>(() => source.CreatePerspectiveCrop(new RotatedRect(PointF.Empty, SizeF.Empty, 0)));
    }

    [Fact]
    public void QuadrilateralDetection_FiltersAndOrdersCandidatesWithoutChangingSource()
    {
        using var binary = EmguCvExtensions.InitMat(new Size(160, 100));
        CvInvoke.Rectangle(binary, new Rectangle(10, 10, 40, 30), new MCvScalar(255), -1);
        CvInvoke.Rectangle(binary, new Rectangle(80, 10, 60, 60), new MCvScalar(255), -1);
        var before = binary.ToArray();
        var candidates = binary.FindQuadrilaterals(minimumArea: 100);
        Assert.Equal(2, candidates.Length);
        Assert.Equal(new PointF(80, 10), candidates[0][0]);
        Assert.Equal(new PointF(10, 10), candidates[1][0]);
        Assert.Single(binary.FindQuadrilaterals(minimumArea: 2000));
        Assert.Single(binary.FindQuadrilaterals(maximumArea: 2000));
        Assert.Empty(binary.FindQuadrilaterals(minimumArea: 10000));
        Assert.Equal(before, binary.ToArray());
        Assert.Throws<ArgumentOutOfRangeException>(() => binary.FindQuadrilaterals(approximationEpsilon: 0));
    }

    [Fact]
    public void BooleanMask_NormalizationAndCopy_UseCanonicalValues()
    {
        using var bytes = new Mat(2, 3, DepthType.Cv8U, 1);
        new byte[] { 0, 1, 255, 128, 0, 2 }.CopyTo(bytes.GetSpan<byte>());
        using var boolean = bytes.ToBooleanMask();
        Assert.Equal(DepthType.CvBool, boolean.Depth);
        Assert.Equal(new byte[] { 0, 1, 1, 1, 0, 1 }, boolean.ToArray());
        using var restored = boolean.ToByteMask();
        Assert.Equal(new byte[] { 0, 255, 255, 255, 0, 255 }, restored.ToArray());
        using var source = EmguCvExtensions.InitMat(bytes.Size);
        source.SetTo(new MCvScalar(42));
        using var destination = EmguCvExtensions.InitMat(bytes.Size);
        source.CopyTo(destination, boolean);
        Assert.Equal(new byte[] { 0, 42, 42, 42, 0, 42 }, destination.ToArray());
        Assert.Equal(new byte[] { 0, 1, 255, 128, 0, 2 }, bytes.ToArray());
    }

    [Fact]
    public void BooleanMask_CreationAndLogicalOperations_MatchTruthTables()
    {
        using var a = EmguCvExtensions.CreateBooleanMask(new bool[] { false, false, true, true }, new Size(4, 1));
        using var b = EmguCvExtensions.CreateBooleanMask(new bool[] { false, true, false, true }, new Size(4, 1));
        using var and = a.MaskAnd(b); using var or = a.MaskOr(b); using var xor = a.MaskXor(b); using var not = a.MaskNot();
        Assert.Equal(new byte[] { 0, 0, 0, 1 }, and.ToArray());
        Assert.Equal(new byte[] { 0, 1, 1, 1 }, or.ToArray());
        Assert.Equal(new byte[] { 0, 1, 1, 0 }, xor.ToArray());
        Assert.Equal(new byte[] { 1, 1, 0, 0 }, not.ToArray());
        using var filled = EmguCvExtensions.CreateBooleanMask(new Size(4, 1), true);
        Assert.Equal(new byte[] { 1, 1, 1, 1 }, filled.ToArray());
        Assert.Equal(new byte[] { 0, 0, 1, 1 }, a.ToArray());
    }

    [Fact]
    public void BooleanMask_MixedNoncontinuousRois_NormalizeNonzeroValues()
    {
        using var source = EmguCvExtensions.InitMat(new Size(5, 3));
        source.SetTo(new MCvScalar(128));
        source.GetSpan2D<byte>()[1, 2] = 0;
        using var roi = new Mat(source, new Rectangle(1, 0, 3, 3));
        using var boolean = EmguCvExtensions.CreateBooleanMask(roi.Size, true);
        using var result = roi.MaskAnd(boolean);
        Assert.Equal(new byte[] { 1, 1, 1, 1, 0, 1, 1, 1, 1 }, result.ToArray());
        using var converted = roi.ToBooleanMask();
        Assert.Equal(result.ToArray(), converted.ToArray());
    }

    [Fact]
    public void BooleanMask_InvalidShapeTypeAndValueCount_AreRejected()
    {
        using var color = new Mat(2, 2, DepthType.Cv8U, 3);
        using var depth = new Mat(2, 2, DepthType.Cv16U, 1);
        using var a = EmguCvExtensions.CreateBooleanMask(new Size(2, 2));
        using var b = EmguCvExtensions.CreateBooleanMask(new Size(3, 2));
        Assert.Throws<ArgumentException>(() => color.ToBooleanMask());
        Assert.Throws<ArgumentException>(() => depth.MaskNot());
        Assert.Throws<ArgumentException>(() => a.MaskAnd(b));
        Assert.Throws<ArgumentException>(() => EmguCvExtensions.CreateBooleanMask(new bool[] { true }, new Size(2, 2)));
    }

    [Theory]
    [InlineData(DepthType.Cv8U, DepthType.Cv8U)]
    [InlineData(DepthType.Cv8U, DepthType.CvBool)]
    [InlineData(DepthType.CvBool, DepthType.Cv8U)]
    [InlineData(DepthType.CvBool, DepthType.CvBool)]
    public void BooleanMask_AllBytePairs_PreserveNonzeroLogicalSemantics(DepthType leftDepth, DepthType rightDepth)
    {
        using var left = new Mat(256, 256, leftDepth, 1);
        using var right = new Mat(256, 256, rightDepth, 1);
        var leftBytes = left.GetSpan<byte>();
        var rightBytes = right.GetSpan<byte>();
        for (var a = 0; a < 256; a++)
        for (var b = 0; b < 256; b++)
        {
            leftBytes[a * 256 + b] = (byte)a;
            rightBytes[a * 256 + b] = (byte)b;
        }
        using var and = left.MaskAnd(right);
        using var or = left.MaskOr(right);
        using var xor = left.MaskXor(right);
        using var not = left.MaskNot();
        using var boolean = left.ToBooleanMask();
        using var bytes = left.ToByteMask();
        Assert.Equal(DepthType.CvBool, and.Depth);
        Assert.Equal(DepthType.CvBool, or.Depth);
        Assert.Equal(DepthType.CvBool, xor.Depth);
        Assert.Equal(DepthType.CvBool, not.Depth);
        Assert.Equal(DepthType.CvBool, boolean.Depth);
        Assert.Equal(DepthType.Cv8U, bytes.Depth);
        var andData = and.GetSpan<byte>(); var orData = or.GetSpan<byte>(); var xorData = xor.GetSpan<byte>();
        var notData = not.GetSpan<byte>(); var boolData = boolean.GetSpan<byte>(); var byteData = bytes.GetSpan<byte>();
        for (var a = 0; a < 256; a++)
        for (var b = 0; b < 256; b++)
        {
            var i = a * 256 + b;
            Assert.Equal(a != 0 && b != 0 ? (byte)1 : (byte)0, andData[i]);
            Assert.Equal(a != 0 || b != 0 ? (byte)1 : (byte)0, orData[i]);
            Assert.Equal((a != 0) != (b != 0) ? (byte)1 : (byte)0, xorData[i]);
            Assert.Equal(a == 0 ? (byte)1 : (byte)0, notData[i]);
            Assert.Equal(a != 0 ? (byte)1 : (byte)0, boolData[i]);
            Assert.Equal(a != 0 ? (byte)255 : (byte)0, byteData[i]);
            Assert.Equal((byte)a, leftBytes[i]);
            Assert.Equal((byte)b, rightBytes[i]);
        }
    }
    private static void AssertPointClose(PointF expected, PointF actual)
    {
        const float tolerance = 0.0001f;
        Assert.InRange(actual.X, expected.X - tolerance, expected.X + tolerance);
        Assert.InRange(actual.Y, expected.Y - tolerance, expected.Y + tolerance);
    }
}