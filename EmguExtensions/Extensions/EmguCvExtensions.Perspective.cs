using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace EmguExtensions;

public static partial class EmguCvExtensions
{
    extension(Mat src)
    {
        /// <summary>
        /// Rectifies a convex quadrilateral into a new image. The caller owns the returned Mat.
        /// </summary>
        /// <param name="corners">Four finite points ordered top-left, top-right, bottom-right, bottom-left.</param>
        /// <param name="outputSize">Output dimensions, each at least two pixels. If empty, opposing edge lengths determine the dimensions.</param>
        /// <param name="interpolation">Interpolation used when sampling the source.</param>
        /// <param name="borderType">How to sample points outside the source image.</param>
        /// <param name="borderValue">Value used for a constant border.</param>
        /// <returns>A rectified image preserving the source depth and channels.</returns>
        /// <exception cref="ArgumentException">The source is empty or the corners do not form a strictly convex quadrilateral.</exception>
        /// <exception cref="ArgumentOutOfRangeException">The output dimensions are invalid.</exception>
        public Mat CreatePerspectiveCrop(
            ReadOnlySpan<PointF> corners,
            Size outputSize = default,
            Inter interpolation = Inter.Linear,
            BorderType borderType = BorderType.Constant,
            MCvScalar borderValue = default)
        {
            ArgumentNullException.ThrowIfNull(src);
            if (src.IsEmpty)
                throw new ArgumentException("The source image must not be empty.", nameof(src));
            outputSize = GetPerspectiveCropSize(corners, outputSize);
            using var transform = CreatePerspectiveTransform(corners, outputSize);
            var result = new Mat();
            try
            {
                CvInvoke.WarpPerspective(src, result, transform, outputSize, interpolation, Warp.Default, borderType, borderValue);
                return result;
            }
            catch
            {
                result.Dispose();
                throw;
            }
        }
    }

    /// <summary>Orders four unordered convex corners clockwise in image coordinates.</summary>
    /// <remarks>The first point has the smallest X+Y; ties use the smaller Y, then X. Diamond orientation is therefore deterministic.</remarks>
    /// <param name="corners">Four distinct finite corners of a convex quadrilateral.</param>
    /// <returns>A new array in top-left, top-right, bottom-right, bottom-left perimeter order.</returns>
    public static PointF[] OrderPerspectiveCorners(ReadOnlySpan<PointF> corners)
    {
        ValidatePerspectiveCornerCoordinates(corners);
        var ordered = corners.ToArray();
        var centerX = ordered.Sum(p => (double)p.X) / 4;
        var centerY = ordered.Sum(p => (double)p.Y) / 4;
        Array.Sort(ordered, (a, b) => Math.Atan2(a.Y - centerY, a.X - centerX)
            .CompareTo(Math.Atan2(b.Y - centerY, b.X - centerX)));
        ValidatePerspectiveCorners(ordered);
        var start = 0;
        for (var i = 1; i < 4; i++)
        {
            var sum = (double)ordered[i].X + ordered[i].Y;
            var bestSum = (double)ordered[start].X + ordered[start].Y;
            if (sum < bestSum || (sum == bestSum && (ordered[i].Y < ordered[start].Y
                || (ordered[i].Y == ordered[start].Y && ordered[i].X < ordered[start].X))))
                start = i;
        }
        return [ordered[start], ordered[(start + 1) % 4], ordered[(start + 2) % 4], ordered[(start + 3) % 4]];
    }

    /// <summary>Validates ordered corners and resolves the crop's output dimensions.</summary>
    /// <param name="corners">Four strictly convex corners in perimeter order.</param>
    /// <param name="outputSize">Explicit dimensions of at least two pixels each, or empty to infer them.</param>
    /// <returns>The explicit size, or rounded-up maximum opposing edge lengths plus one pixel.</returns>
    public static Size GetPerspectiveCropSize(ReadOnlySpan<PointF> corners, Size outputSize = default)
    {
        ValidatePerspectiveCorners(corners);
        if (outputSize == Size.Empty)
        {
            var width = Math.Max(PerspectiveEdgeLength(corners[0], corners[1]), PerspectiveEdgeLength(corners[3], corners[2]));
            var height = Math.Max(PerspectiveEdgeLength(corners[0], corners[3]), PerspectiveEdgeLength(corners[1], corners[2]));
            if (width > int.MaxValue - 1 || height > int.MaxValue - 1)
                throw new ArgumentOutOfRangeException(nameof(corners), "The inferred image size is too large.");
            outputSize = new Size((int)Math.Ceiling(width) + 1, (int)Math.Ceiling(height) + 1);
        }
        if (outputSize.Width < 2 || outputSize.Height < 2)
            throw new ArgumentOutOfRangeException(nameof(outputSize), "Both output dimensions must be at least two pixels.");
        return outputSize;
    }

    /// <summary>Creates a caller-owned 3x3 transform from source coordinates to crop coordinates, or its inverse.</summary>
    /// <param name="corners">Four ordered convex source corners.</param>
    /// <param name="outputSize">Explicit crop dimensions, or empty to infer them.</param>
    /// <param name="inverse">True maps from crop coordinates back to the source.</param>
    /// <returns>A caller-owned double-precision transform Mat.</returns>
    public static Mat CreatePerspectiveTransform(ReadOnlySpan<PointF> corners, Size outputSize = default, bool inverse = false)
    {
        outputSize = GetPerspectiveCropSize(corners, outputSize);
        PointF[] destination = [new(0, 0), new(outputSize.Width - 1, 0),
            new(outputSize.Width - 1, outputSize.Height - 1), new(0, outputSize.Height - 1)];
        var source = corners.ToArray();
        return inverse ? CvInvoke.GetPerspectiveTransform(destination, source) : CvInvoke.GetPerspectiveTransform(source, destination);
    }

    extension(Mat src)
    {
        /// <summary>Rectifies a quadrilateral into a reusable destination, allocating it if its shape or type differs.</summary>
        /// <remarks>Overlapping source/destination buffers use a temporary image. Neither Mat is disposed.</remarks>
        /// <param name="destination">Caller-owned destination, which may be empty.</param>
        /// <param name="corners">Four ordered convex corners.</param>
        /// <param name="outputSize">Explicit dimensions, or empty to infer them.</param>
        /// <param name="interpolation">Sampling interpolation.</param>
        /// <param name="borderType">Outside-image sampling behavior.</param>
        /// <param name="borderValue">Constant outside-image value.</param>
        public void PerspectiveCrop(Mat destination, ReadOnlySpan<PointF> corners, Size outputSize = default,
            Inter interpolation = Inter.Linear, BorderType borderType = BorderType.Constant, MCvScalar borderValue = default)
        {
            ArgumentNullException.ThrowIfNull(src);
            ArgumentNullException.ThrowIfNull(destination);
            if (src.IsEmpty) throw new ArgumentException("The source image must not be empty.", nameof(src));
            outputSize = GetPerspectiveCropSize(corners, outputSize);
            using var transform = CreatePerspectiveTransform(corners, outputSize);
            if (PerspectiveBuffersOverlap(src, destination))
            {
                using var temporary = new Mat();
                CvInvoke.WarpPerspective(src, temporary, transform, outputSize, interpolation, Warp.Default, borderType, borderValue);
                temporary.CopyTo(destination);
            }
            else CvInvoke.WarpPerspective(src, destination, transform, outputSize, interpolation, Warp.Default, borderType, borderValue);
        }

        /// <summary>Rectifies a minimum-area rotated rectangle into a caller-owned image.</summary>
        /// <remarks>Vertices are deterministically ordered; the first edge determines the crop's horizontal axis.</remarks>
        /// <param name="rectangle">A finite rotated rectangle with positive dimensions.</param>
        /// <param name="outputSize">Explicit output dimensions, or empty to infer them.</param>
        /// <param name="interpolation">Sampling interpolation.</param>
        /// <param name="borderType">Outside-image sampling behavior.</param>
        /// <param name="borderValue">Constant outside-image value.</param>
        /// <returns>A caller-owned rectified image.</returns>
        public Mat CreatePerspectiveCrop(RotatedRect rectangle, Size outputSize = default,
            Inter interpolation = Inter.Linear, BorderType borderType = BorderType.Constant, MCvScalar borderValue = default)
        {
            if (!float.IsFinite(rectangle.Size.Width) || !float.IsFinite(rectangle.Size.Height)
                || rectangle.Size.Width <= 0 || rectangle.Size.Height <= 0 || !float.IsFinite(rectangle.Angle))
                throw new ArgumentOutOfRangeException(nameof(rectangle));
            return src.CreatePerspectiveCrop(OrderPerspectiveCorners(rectangle.GetVertices()), outputSize, interpolation, borderType, borderValue);
        }

        /// <summary>Maps finite points through this 3x3 double-precision perspective transform.</summary>
        /// <param name="points">Source points; an empty span produces an empty array.</param>
        /// <returns>New points in the transform's destination coordinate system.</returns>
        /// <exception cref="ArgumentException">The matrix is not a finite 3x3 double-precision single-channel transform.</exception>
        /// <exception cref="ArgumentOutOfRangeException">A point maps to infinity or outside the finite float coordinate range.</exception>
        public PointF[] MapPerspectivePoints(ReadOnlySpan<PointF> points)
        {
            ArgumentNullException.ThrowIfNull(src);
            if (src.Rows != 3 || src.Cols != 3 || src.Depth != DepthType.Cv64F || src.NumberOfChannels != 1)
                throw new ArgumentException("A 3x3 double-precision single-channel transform is required.", nameof(src));
            var matrix = src.GetReadOnlySpan2D<double>();
            for (var y = 0; y < 3; y++)
                for (var x = 0; x < 3; x++)
                    if (!double.IsFinite(matrix[y, x])) throw new ArgumentException("Transform coefficients must be finite.", nameof(src));
            var mapped = new PointF[points.Length];
            for (var i = 0; i < points.Length; i++)
            {
                var p = points[i];
                if (!float.IsFinite(p.X) || !float.IsFinite(p.Y)) throw new ArgumentOutOfRangeException(nameof(points));
                var divisor = matrix[2, 0] * p.X + matrix[2, 1] * p.Y + matrix[2, 2];
                var x = (matrix[0, 0] * p.X + matrix[0, 1] * p.Y + matrix[0, 2]) / divisor;
                var y = (matrix[1, 0] * p.X + matrix[1, 1] * p.Y + matrix[1, 2]) / divisor;
                if (!double.IsFinite(x) || !double.IsFinite(y) || Math.Abs(x) > float.MaxValue || Math.Abs(y) > float.MaxValue)
                    throw new ArgumentOutOfRangeException(nameof(points), "A point maps to infinity or outside the finite coordinate range.");
                mapped[i] = new PointF((float)x, (float)y);
            }
            return mapped;
        }

        /// <summary>Finds convex four-corner contours in a nonempty single-channel 8-bit binary image.</summary>
        /// <remarks>The source is preserved. Results contain managed ordered corner arrays, largest approximated area first.</remarks>
        /// <param name="minimumArea">Inclusive minimum approximated polygon area in square pixels.</param>
        /// <param name="maximumArea">Inclusive maximum approximated polygon area in square pixels.</param>
        /// <param name="approximationEpsilon">Approximation tolerance as a positive fraction of contour perimeter, less than one.</param>
        /// <param name="retrievalMode">Contour retrieval mode, external contours by default.</param>
        /// <returns>Ordered convex quadrilaterals; no match produces an empty array.</returns>
        public PointF[][] FindQuadrilaterals(double minimumArea = 0, double maximumArea = double.MaxValue,
            double approximationEpsilon = 0.02, RetrType retrievalMode = RetrType.External)
        {
            ArgumentNullException.ThrowIfNull(src);
            if (src.IsEmpty || src.Depth != DepthType.Cv8U || src.NumberOfChannels != 1)
                throw new ArgumentException("A nonempty single-channel 8-bit binary image is required.", nameof(src));
            if (!double.IsFinite(minimumArea) || minimumArea < 0) throw new ArgumentOutOfRangeException(nameof(minimumArea));
            if (!double.IsFinite(maximumArea) || maximumArea < minimumArea) throw new ArgumentOutOfRangeException(nameof(maximumArea));
            if (!double.IsFinite(approximationEpsilon) || approximationEpsilon <= 0 || approximationEpsilon >= 1)
                throw new ArgumentOutOfRangeException(nameof(approximationEpsilon));
            if (retrievalMode is not (RetrType.External or RetrType.List or RetrType.Ccomp or RetrType.Tree))
                throw new ArgumentOutOfRangeException(nameof(retrievalMode));
            using var contours = new Emgu.CV.Util.VectorOfVectorOfPoint();
            CvInvoke.FindContours(src, contours, null, retrievalMode, ChainApproxMethod.ChainApproxSimple);
            var candidates = new List<(double Area, PointF[] Corners)>();
            for (var i = 0; i < contours.Size; i++)
            {
                using var contour = contours[i];
                using var polygon = new Emgu.CV.Util.VectorOfPoint();
                CvInvoke.ApproxPolyDP(contour, polygon, approximationEpsilon * CvInvoke.ArcLength(contour, true), true);
                if (polygon.Size != 4 || !CvInvoke.IsContourConvex(polygon)) continue;
                var area = CvInvoke.ContourArea(polygon);
                if (area <= 0 || area < minimumArea || area > maximumArea) continue;
                var corners = polygon.ToArray().Select(p => new PointF(p.X, p.Y)).ToArray();
                candidates.Add((area, OrderPerspectiveCorners(corners)));
            }
            return candidates.OrderByDescending(c => c.Area).Select(c => c.Corners).ToArray();
        }
    }

    private static void ValidatePerspectiveCornerCoordinates(ReadOnlySpan<PointF> corners)
    {
        if (corners.Length != 4) throw new ArgumentException("Exactly four corners are required.", nameof(corners));
        foreach (var p in corners)
            if (!float.IsFinite(p.X) || !float.IsFinite(p.Y))
                throw new ArgumentException("All corner coordinates must be finite.", nameof(corners));
    }

    private static void ValidatePerspectiveCorners(ReadOnlySpan<PointF> corners)
    {
        ValidatePerspectiveCornerCoordinates(corners);
        double previousCross = 0;
        for (var i = 0; i < 4; i++)
        {
            var a = corners[i]; var b = corners[(i + 1) % 4]; var c = corners[(i + 2) % 4];
            var cross = ((double)b.X - a.X) * ((double)c.Y - b.Y) - ((double)b.Y - a.Y) * ((double)c.X - b.X);
            if (cross == 0 || (i > 0 && Math.Sign(cross) != Math.Sign(previousCross)))
                throw new ArgumentException("Corners must form a strictly convex quadrilateral in perimeter order.", nameof(corners));
            previousCross = cross;
        }
    }

    private static bool PerspectiveBuffersOverlap(Mat a, Mat b)
    {
        if (b.IsEmpty) return false;
        var startA = (ulong)a.DataPointer.ToInt64();
        var startB = (ulong)b.DataPointer.ToInt64();
        var endA = startA + (ulong)((long)(a.Height - 1) * a.Step + a.RealStep);
        var endB = startB + (ulong)((long)(b.Height - 1) * b.Step + b.RealStep);
        return startA < endB && startB < endA;
    }
    private static double PerspectiveEdgeLength(PointF a, PointF b)
    {
        var x = (double)b.X - a.X;
        var y = (double)b.Y - a.Y;
        return Math.Sqrt(x * x + y * y);
    }
}