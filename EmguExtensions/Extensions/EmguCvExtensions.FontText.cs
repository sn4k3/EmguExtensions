using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using CvFontFace = Emgu.CV.FontFace;
using CvRange = Emgu.CV.Structure.Range;

namespace EmguExtensions;

public static partial class EmguCvExtensions
{
    /// <summary>Measures Unicode text, including explicit line breaks and optional wrapping, in pixels.</summary>
    /// <remarks>The caller owns the font. Alignment applies to the whole text block; glyph coverage depends on the font.</remarks>
    /// <param name="text">Text to measure.</param>
    /// <param name="fontFace">A caller-owned embedded or custom font.</param>
    /// <param name="fontSize">Positive font size in pixels.</param>
    /// <param name="weight">Zero uses the font's current weight; otherwise use 100 through 1000.</param>
    /// <param name="lineAlignment">Text block alignment and trimming behavior.</param>
    /// <param name="wrapWidth">Positive wrapping width in pixels, or zero to disable wrapping.</param>
    /// <returns>The bounding size of the rendered text.</returns>
    public static Size GetTextSizeExtended(
        string text, CvFontFace fontFace, int fontSize, int weight = 0,
        PutTextLineAlignment lineAlignment = default, int wrapWidth = 0)
    {
        var prepared = PrepareFontText(text, fontFace, fontSize, weight, lineAlignment, wrapWidth);
        return MeasureFontText(prepared, fontFace, fontSize, weight, wrapWidth).Size;
    }

    extension(Mat src)
    {
        /// <summary>Draws Unicode text with a caller-owned font, line breaks, block alignment, and optional wrapping.</summary>
        /// <remarks>The origin is the first line's baseline. Center and right alignment anchor the whole block at the origin's X coordinate.</remarks>
        /// <param name="text">Text to draw.</param>
        /// <param name="org">Baseline origin in image coordinates.</param>
        /// <param name="fontFace">A caller-owned font; this method does not dispose it.</param>
        /// <param name="fontSize">Positive font size in pixels.</param>
        /// <param name="color">Text color in the source image's channel order.</param>
        /// <param name="weight">Zero uses the current font weight, or 100 through 1000.</param>
        /// <param name="lineAlignment">Whole-block alignment and trimming behavior.</param>
        /// <param name="wrapWidth">Wrapping width, or zero to disable wrapping.</param>
        public void PutTextExtended(
            string text, Point org, CvFontFace fontFace, int fontSize, MCvScalar color,
            int weight = 0, PutTextLineAlignment lineAlignment = default, int wrapWidth = 0)
        {
            ArgumentNullException.ThrowIfNull(src);
            var prepared = PrepareFontText(text, fontFace, fontSize, weight, lineAlignment, wrapWidth);
            if (src.IsEmpty || prepared.Length == 0)
                return;
            var bounds = MeasureFontText(prepared, fontFace, fontSize, weight, wrapWidth);
            var offsetX = FontTextAlignmentOffset(bounds, lineAlignment);
            var drawOrg = new Point(checked(org.X + offsetX), org.Y);
            DrawFontText(src, prepared, drawOrg, fontFace, fontSize, color, weight, wrapWidth);
        }

        /// <summary>Draws Unicode text rotated clockwise around its baseline origin, preserving the surrounding image.</summary>
        /// <param name="text">Text to draw.</param>
        /// <param name="org">Baseline origin and rotation pivot.</param>
        /// <param name="fontFace">A caller-owned font.</param>
        /// <param name="fontSize">Positive font size in pixels.</param>
        /// <param name="color">Text color.</param>
        /// <param name="weight">Zero uses the current font weight, or 100 through 1000.</param>
        /// <param name="lineAlignment">Whole-block alignment and trimming behavior.</param>
        /// <param name="wrapWidth">Wrapping width, or zero to disable wrapping.</param>
        /// <param name="angle">Clockwise rotation in degrees.</param>
        public void PutTextRotated(
            string text, Point org, CvFontFace fontFace, int fontSize, MCvScalar color,
            int weight = 0, PutTextLineAlignment lineAlignment = default, int wrapWidth = 0, double angle = 0)
        {
            ArgumentNullException.ThrowIfNull(src);
            if (!double.IsFinite(angle))
                throw new ArgumentOutOfRangeException(nameof(angle));
            if (angle % 360 == 0)
            {
                src.PutTextExtended(text, org, fontFace, fontSize, color, weight, lineAlignment, wrapWidth);
                return;
            }

            var prepared = PrepareFontText(text, fontFace, fontSize, weight, lineAlignment, wrapWidth);
            if (src.IsEmpty || prepared.Length == 0)
                return;

            var bounds = MeasureFontText(prepared, fontFace, fontSize, weight, wrapWidth);
            if (bounds.Width <= 0 || bounds.Height <= 0)
                return;
            const int margin = 2;
            var layerSize = new Size(checked(bounds.Width + margin * 2), checked(bounds.Height + margin * 2));
            var alignmentOffset = FontTextAlignmentOffset(bounds, lineAlignment);
            var layerOrg = new Point(checked(margin - bounds.X - alignmentOffset), checked(margin - bounds.Y));
            DrawRotatedLayer(src, org, layerOrg, layerSize, angle, color,
                (target, drawColor) => target.PutTextExtended(text, layerOrg, fontFace, fontSize, drawColor, weight, lineAlignment, wrapWidth));
        }

        /// <summary>Fits and draws wrapped Unicode text inside a rectangle using the largest font size in the requested range.</summary>
        /// <remarks>Text is clipped to the rectangle. The caller owns the font. Nothing is drawn if the minimum size does not fit.</remarks>
        /// <param name="text">Text to draw.</param>
        /// <param name="rectangle">A positive rectangle wholly contained in the source image.</param>
        /// <param name="fontFace">A caller-owned font.</param>
        /// <param name="color">Text color.</param>
        /// <param name="maximumFontSize">Largest permitted font size in pixels.</param>
        /// <param name="minimumFontSize">Smallest permitted font size in pixels.</param>
        /// <param name="weight">Zero uses the current font weight, or 100 through 1000.</param>
        /// <param name="lineAlignment">Whole-block horizontal alignment.</param>
        /// <returns>The selected font size, or zero if the text is empty or cannot fit.</returns>
        public int PutTextInRectangle(
            string text, Rectangle rectangle, CvFontFace fontFace, MCvScalar color,
            int maximumFontSize = 64, int minimumFontSize = 1, int weight = 0,
            PutTextLineAlignment lineAlignment = default)
        {
            ArgumentNullException.ThrowIfNull(src);
            if (rectangle.Width <= 0 || rectangle.Height <= 0 || rectangle.X < 0 || rectangle.Y < 0
                || (long)rectangle.X + rectangle.Width > src.Width || (long)rectangle.Y + rectangle.Height > src.Height)
                throw new ArgumentOutOfRangeException(nameof(rectangle), "The rectangle must be positive and inside the image.");
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(minimumFontSize);
            ArgumentOutOfRangeException.ThrowIfLessThan(maximumFontSize, minimumFontSize);
            var prepared = PrepareFontText(text, fontFace, minimumFontSize, weight, lineAlignment, rectangle.Width);
            if (prepared.Length == 0)
                return 0;
            var selected = 0;
            var low = minimumFontSize;
            var high = maximumFontSize;
            while (low <= high)
            {
                var size = low + (high - low) / 2;
                var measured = MeasureFontText(prepared, fontFace, size, weight, rectangle.Width);
                if (measured.Width <= rectangle.Width && measured.Height <= rectangle.Height)
                {
                    selected = size;
                    if (size == high) break;
                    low = size + 1;
                }
                else high = size - 1;
            }
            if (selected == 0)
                return 0;
            var bounds = MeasureFontText(prepared, fontFace, selected, weight, rectangle.Width);
            using var destination = new Mat(src, rectangle);
            var x = lineAlignment switch
            {
                PutTextLineAlignment.Center => (rectangle.Width - bounds.Width) / 2,
                PutTextLineAlignment.Right => rectangle.Width - bounds.Width,
                _ => 0
            };
            DrawFontText(destination, prepared, new Point(x - bounds.X, -bounds.Y), fontFace, selected, color, weight, rectangle.Width);
            return selected;
        }
    }

    private static string PrepareFontText(string text, CvFontFace fontFace, int fontSize, int weight,
        PutTextLineAlignment alignment, int wrapWidth)
    {
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(fontFace);
        ObjectDisposedException.ThrowIf(fontFace.Ptr == IntPtr.Zero, fontFace);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(fontSize);
        ArgumentOutOfRangeException.ThrowIfNegative(wrapWidth);
        if (weight != 0 && (weight < 100 || weight > 1000))
            throw new ArgumentOutOfRangeException(nameof(weight), "Weight must be zero or between 100 and 1000.");
        if (!Enum.IsDefined(alignment))
            throw new ArgumentOutOfRangeException(nameof(alignment));
        return string.Join('\n', text.Replace("\r\n", "\n").Replace('\r', '\n')
            .TrimEnd('\n', ' ').Split('\n').Select(line => PutTextLineAlignmentTrim(line, alignment)));
    }

    private static Rectangle MeasureFontText(string text, CvFontFace fontFace, int fontSize, int weight, int wrapWidth)
    {
        if (text.Length == 0)
            return Rectangle.Empty;
        return CvInvoke.GetTextSize(wrapWidth > 0 ? new Size(wrapWidth, int.MaxValue) : Size.Empty, text, Point.Empty, fontFace, fontSize, weight,
            wrapWidth > 0 ? PutTextFlags.Wrap : PutTextFlags.AlignLeft,
            wrapWidth > 0 ? new CvRange(0, wrapWidth) : default);
    }

    private static int FontTextAlignmentOffset(Rectangle bounds, PutTextLineAlignment alignment) => alignment switch
    {
        PutTextLineAlignment.Center => -bounds.X - bounds.Width / 2,
        PutTextLineAlignment.Right => -bounds.X - bounds.Width,
        _ => 0
    };

    private static void DrawFontText(Mat src, string text, Point org, CvFontFace fontFace, int fontSize,
        MCvScalar color, int weight, int wrapWidth)
    {
        CvInvoke.PutText(src, text, org, color, fontFace, fontSize, weight,
            wrapWidth > 0 ? PutTextFlags.Wrap : PutTextFlags.AlignLeft,
            wrapWidth > 0 ? new CvRange(org.X, checked(org.X + wrapWidth)) : default);
    }

    /// <summary>Renders text on a layer around its origin, rotates the layer into the source, and copies it through its mask.</summary>
    /// <remarks>The layer is rotated clockwise around <paramref name="org"/>. Nothing is drawn when the layer lies entirely outside the source.</remarks>
    private static void DrawRotatedLayer(Mat src, Point org, Point layerOrg, Size layerSize, double angle,
        MCvScalar color, Action<Mat, MCvScalar> drawText)
    {
        // Forward transform layer -> src: rotates around the text origin, then moves the origin into place
        using var forward = new Mat(2, 3, DepthType.Cv64F, 1);
        CvInvoke.GetRotationMatrix2D(layerOrg, -angle, 1, forward);
        var matrix = forward.GetSpan2D<double>();
        matrix[0, 2] += org.X - layerOrg.X;
        matrix[1, 2] += org.Y - layerOrg.Y;

        // Bounds of the rotated layer inside src
        double minX = double.MaxValue, minY = double.MaxValue;
        double maxX = double.MinValue, maxY = double.MinValue;
        for (var i = 0; i < 4; i++)
        {
            var x = (i & 1) == 0 ? 0 : layerSize.Width;
            var y = (i & 2) == 0 ? 0 : layerSize.Height;
            var tx = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2];
            var ty = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2];
            minX = Math.Min(minX, tx); minY = Math.Min(minY, ty);
            maxX = Math.Max(maxX, tx); maxY = Math.Max(maxY, ty);
        }
        // Clamp in floating point before converting, including for origins far outside the image.
        minX = Math.Max(0, Math.Floor(minX) - 1);
        minY = Math.Max(0, Math.Floor(minY) - 1);
        maxX = Math.Min(src.Width, Math.Ceiling(maxX) + 1);
        maxY = Math.Min(src.Height, Math.Ceiling(maxY) + 1);
        if (maxX <= minX || maxY <= minY)
            return;
        var region = Rectangle.FromLTRB((int)minX, (int)minY, (int)maxX, (int)maxY);

        // Background of the layer is the (inverse rotated) image, so the anti-aliased text edges blend with the real background
        using var layer = new Mat();
        CvInvoke.WarpAffine(src, layer, forward, layerSize, Inter.Linear, Warp.InverseMap, BorderType.Replicate);
        drawText(layer, color);
        using var mask = InitMat(layerSize);
        drawText(mask, WhiteColor);

        // Rotate the layer and its mask back into the region of src that they cover, and copy through the mask
        matrix[0, 2] -= region.X;
        matrix[1, 2] -= region.Y;
        using var regionLayer = new Mat();
        using var regionMask = new Mat();
        CvInvoke.WarpAffine(layer, regionLayer, forward, region.Size, Inter.Linear, Warp.Default, BorderType.Replicate);
        CvInvoke.WarpAffine(mask, regionMask, forward, region.Size);
        using var destination = new Mat(src, region);
        regionLayer.CopyTo(destination, regionMask);
    }
}