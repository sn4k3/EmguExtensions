using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using CvFontFace = Emgu.CV.FontFace;

namespace EmguExtensions.Tests;

public class UnitTestOpenCv5Features
{
    [Fact]
    public void PerspectiveCrop_AxisAligned_InferredSizeMatchesRoiAndPreservesSource()
    {
        using var source = EmguCvExtensions.InitMat(new Size(9, 7), 3);
        for (var y = 0; y < source.Height; y++)
        {
            var row = source.GetRowSpanOfBytes(y);
            for (var x = 0; x < row.Length; x++) row[x] = (byte)(y * 20 + x);
        }
        var before = source.ToArray();
        PointF[] corners = [new(2, 1), new(6, 1), new(6, 4), new(2, 4)];
        using var crop = source.CreatePerspectiveCrop(corners, interpolation: Inter.Nearest);
        using var expected = new Mat(source, new Rectangle(2, 1, 5, 4));
        Assert.Equal(expected.Size, crop.Size);
        Assert.Equal(source.Depth, crop.Depth);
        Assert.Equal(source.NumberOfChannels, crop.NumberOfChannels);
        Assert.Equal(expected.ToArray(), crop.ToArray());
        Assert.Equal(before, source.ToArray());
    }

    [Fact]
    public void PerspectiveCrop_Trapezoid_MapsAllFourCorners()
    {
        using var source = EmguCvExtensions.InitMat(new Size(10, 10));
        PointF[] corners = [new(2, 1), new(7, 1), new(8, 8), new(1, 8)];
        var pixels = source.GetSpan2D<byte>();
        pixels[1, 2] = 40; pixels[1, 7] = 80;
        pixels[8, 8] = 120; pixels[8, 1] = 160;
        using var crop = source.CreatePerspectiveCrop(corners, new Size(6, 8), Inter.Nearest);
        var output = crop.GetReadOnlySpan2D<byte>();
        Assert.Equal(40, output[0, 0]); Assert.Equal(80, output[0, 5]);
        Assert.Equal(120, output[7, 5]); Assert.Equal(160, output[7, 0]);
    }

    [Theory]
    [InlineData(0)] // Wrong number of corners.
    [InlineData(1)] // Crossed perimeter.
    [InlineData(2)] // Collinear corner.
    [InlineData(3)] // Non-finite coordinates.
    [InlineData(4)] // Repeated point.
    public void PerspectiveCrop_InvalidCorners_AreRejected(int scenario)
    {
        using var source = EmguCvExtensions.InitMat(new Size(10, 10));
        PointF[] corners = scenario switch
        {
            0 => [new(0, 0)],
            1 => [new(0, 0), new(5, 5), new(5, 0), new(0, 5)],
            2 => [new(0, 0), new(2, 0), new(5, 0), new(0, 5)],
            3 => [new(float.NaN, 0), new(5, 0), new(5, 5), new(0, 5)],
            _ => [new(0, 0), new(5, 0), new(5, 0), new(0, 5)]
        };
        Assert.Throws<ArgumentException>(() => source.CreatePerspectiveCrop(corners));
    }

    [Fact]
    public void PerspectiveCrop_InvalidOutputSize_IsRejected()
    {
        using var source = EmguCvExtensions.InitMat(new Size(10, 10));
        PointF[] corners = [new(0, 0), new(5, 0), new(5, 5), new(0, 5)];
        Assert.Throws<ArgumentOutOfRangeException>(() => source.CreatePerspectiveCrop(corners, new Size(1, 5)));
    }

    [Fact]
    public void FontText_UnicodeAndLineBreaks_RenderAndFontRemainsReusable()
    {
        using var font = new CvFontFace("uni");
        using var image = EmguCvExtensions.InitMat(new Size(400, 200));
        const string text = "Inspeção concluída\n你好";
        var size = EmguCvExtensions.GetTextSizeExtended(text, font, 24);
        var singleLine = EmguCvExtensions.GetTextSizeExtended("Inspeção concluída", font, 24);
        image.PutTextExtended(text, new Point(20, 45), font, 24, new MCvScalar(255));
        Assert.True(size.Width > 0);
        Assert.True(size.Height > singleLine.Height);
        Assert.True(image.CountNonZero > 0);
        Assert.Equal(size, EmguCvExtensions.GetTextSizeExtended(text, font, 24));
    }

    [Fact]
    public void FontText_Wrap_IncreasesHeight()
    {
        using var font = new CvFontFace("sans");
        const string text = "A long label with several words to wrap";
        var unwrapped = EmguCvExtensions.GetTextSizeExtended(text, font, 24);
        var wrapped = EmguCvExtensions.GetTextSizeExtended(text, font, 24, wrapWidth: 100);
        Assert.True(wrapped.Height > unwrapped.Height, $"Unwrapped: {unwrapped}, wrapped: {wrapped}");
        Assert.True(wrapped.Width < unwrapped.Width, $"Unwrapped: {unwrapped}, wrapped: {wrapped}");
    }

    [Theory]
    [InlineData(PutTextLineAlignment.Center)]
    [InlineData(PutTextLineAlignment.Right)]
    public void FontText_Alignment_ShiftsTheWholeBlock(PutTextLineAlignment alignment)
    {
        using var font = new CvFontFace("sans");
        using var left = EmguCvExtensions.InitMat(new Size(300, 100));
        using var aligned = EmguCvExtensions.InitMat(left.Size);
        const string text = "Label";
        var size = EmguCvExtensions.GetTextSizeExtended(text, font, 24, lineAlignment: alignment);
        left.PutTextExtended(text, new Point(150, 50), font, 24, new MCvScalar(255));
        aligned.PutTextExtended(text, new Point(150, 50), font, 24, new MCvScalar(255), lineAlignment: alignment);
        var leftBounds = CvInvoke.BoundingRectangle(left);
        var alignedBounds = CvInvoke.BoundingRectangle(aligned);
        Assert.True(alignedBounds.Left < leftBounds.Left);
        Assert.Equal(leftBounds.Size, alignedBounds.Size);
        Assert.InRange(alignedBounds.Right, 150 - 24 / 4, alignment == PutTextLineAlignment.Center ? 150 + size.Width / 2 : 150);
    }

    [Fact]
    public void FontText_RotationZero_MatchesOrdinaryRendering()
    {
        using var font = new CvFontFace("sans");
        using var ordinary = EmguCvExtensions.InitMat(new Size(200, 100));
        using var rotated = EmguCvExtensions.InitMat(ordinary.Size);
        ordinary.PutTextExtended("Olá", new Point(60, 50), font, 24, new MCvScalar(255));
        rotated.PutTextRotated("Olá", new Point(60, 50), font, 24, new MCvScalar(255));
        Assert.Equal(ordinary.ToArray(), rotated.ToArray());
    }

    [Fact]
    public void FontText_Rotation90_ChangesAxisAndPreservesDistantBackground()
    {
        using var font = new CvFontFace("sans");
        using var horizontal = EmguCvExtensions.InitMat(new Size(300, 300));
        using var vertical = EmguCvExtensions.InitMat(horizontal.Size);
        horizontal.PutTextRotated("A long label", new Point(100, 100), font, 24, new MCvScalar(255));
        vertical.PutTextRotated("A long label", new Point(100, 100), font, 24, new MCvScalar(255), angle: 90);
        var horizontalBounds = CvInvoke.BoundingRectangle(horizontal);
        var verticalBounds = CvInvoke.BoundingRectangle(vertical);
        Assert.True(horizontalBounds.Width > horizontalBounds.Height);
        Assert.True(verticalBounds.Height > verticalBounds.Width);
        Assert.Equal(0, vertical.GetReadOnlySpan2D<byte>()[299, 299]);
    }

    [Fact]
    public void FontText_FullyOutsideRotation_DoesNotChangeImage()
    {
        using var font = new CvFontFace("sans");
        using var image = EmguCvExtensions.InitMat(new Size(100, 100));
        image.PutTextRotated("Outside", new Point(int.MaxValue, int.MaxValue), font, 24, new MCvScalar(255), angle: 30);
        Assert.Equal(0, image.CountNonZero);
    }

    [Fact]
    public void FontText_FitRectangle_ChoosesSizeAndPreservesOutsidePixels()
    {
        using var font = new CvFontFace("sans");
        using var image = EmguCvExtensions.InitMat(new Size(240, 180));
        image.SetTo(new MCvScalar(50));
        var rectangle = new Rectangle(30, 40, 120, 80);
        const string text = "Inspeção concluída";
        var selected = image.PutTextInRectangle(text, rectangle, font, new MCvScalar(255), maximumFontSize: 48);
        Assert.InRange(selected, 1, 48);
        var size = EmguCvExtensions.GetTextSizeExtended(text, font, selected, wrapWidth: rectangle.Width);
        Assert.True(size.Width <= rectangle.Width && size.Height <= rectangle.Height);
        if (selected < 48)
        {
            var larger = EmguCvExtensions.GetTextSizeExtended(text, font, selected + 1, wrapWidth: rectangle.Width);
            Assert.True(larger.Width > rectangle.Width || larger.Height > rectangle.Height);
        }
        using var changed = new Mat();
        using var background = new ScalarArray(50);
        CvInvoke.Compare(image, background, changed, CmpType.NotEqual);
        Assert.True(changed.CountNonZero > 0);
        var bounds = CvInvoke.BoundingRectangle(changed);
        Assert.True(rectangle.Contains(bounds));
    }

    [Fact]
    public void FontText_CannotFit_DoesNotDraw()
    {
        using var font = new CvFontFace("sans");
        using var image = EmguCvExtensions.InitMat(new Size(100, 100));
        var selected = image.PutTextInRectangle("Large text", new Rectangle(10, 10, 2, 2), font,
            new MCvScalar(255), maximumFontSize: 24, minimumFontSize: 24);
        Assert.Equal(0, selected);
        Assert.Equal(0, image.CountNonZero);
    }

    [Fact]
    public void FontText_AccentedGlyph_IsNotReplacedByQuestionMark()
    {
        using var font = new CvFontFace("uni");
        using var accented = EmguCvExtensions.InitMat(new Size(100, 100));
        using var replacement = EmguCvExtensions.InitMat(accented.Size);
        accented.PutTextExtended("ç", new Point(20, 50), font, 32, new MCvScalar(255));
        replacement.PutTextExtended("?", new Point(20, 50), font, 32, new MCvScalar(255));
        Assert.True(accented.CountNonZero > 0);
        Assert.False(accented.ToArray().SequenceEqual(replacement.ToArray()));
    }

    [Fact]
    public void FontText_EmptyAndDisposedFont_RespectContracts()
    {
        using var image = EmguCvExtensions.InitMat(new Size(100, 100));
        using var font = new CvFontFace("sans");
        Assert.Equal(Size.Empty, EmguCvExtensions.GetTextSizeExtended("", font, 24));
        Assert.Equal(0, image.PutTextInRectangle("", new Rectangle(10, 10, 80, 80), font, new MCvScalar(255)));
        image.PutTextExtended("", Point.Empty, font, 24, new MCvScalar(255));
        Assert.Equal(0, image.CountNonZero);
        font.Dispose();
        Assert.Throws<ObjectDisposedException>(() => EmguCvExtensions.GetTextSizeExtended("Label", font, 24));
    }

    [Fact]
    public void FontText_InvalidArguments_AreRejected()
    {
        using var font = new CvFontFace("sans");
        Assert.Throws<ArgumentOutOfRangeException>(() => EmguCvExtensions.GetTextSizeExtended("Label", font, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => EmguCvExtensions.GetTextSizeExtended("Label", font, 24, weight: 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => EmguCvExtensions.GetTextSizeExtended("Label", font, 24, wrapWidth: -1));
    }
}