using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace EmguExtensions;

public static partial class EmguCvExtensions
{
    /// <summary>Creates a caller-owned single-channel OpenCV boolean mask filled with the requested value.</summary>
    /// <param name="size">Positive mask dimensions.</param>
    /// <param name="value">Initial value of every mask element.</param>
    /// <returns>A mask with canonical byte values zero and one.</returns>
    public static Mat CreateBooleanMask(Size size, bool value = false)
    {
        if (size.Width <= 0 || size.Height <= 0)
            throw new ArgumentOutOfRangeException(nameof(size));
        if (value)
        {
            return Mat.Ones(size.Height, size.Width, DepthType.CvBool, 1);
        }
        else
        {
            return Mat.Zeros(size.Height, size.Width, DepthType.CvBool, 1);
        }
    }

    /// <summary>Copies row-major boolean values into a caller-owned OpenCV boolean mask.</summary>
    /// <param name="values">Exactly width times height values, ordered by rows.</param>
    /// <param name="size">Positive mask dimensions.</param>
    /// <returns>A mask with canonical byte values zero and one.</returns>
    public static Mat CreateBooleanMask(ReadOnlySpan<bool> values, Size size)
    {
        if (size.Width <= 0 || size.Height <= 0)
            throw new ArgumentOutOfRangeException(nameof(size));
        if ((long)size.Width * size.Height != values.Length)
            throw new ArgumentException(
                "The value count must match the mask dimensions.",
                nameof(values)
            );
        var result = new Mat(size, DepthType.CvBool, 1);
        try
        {
            var offset = 0;
            for (var y = 0; y < size.Height; y++)
            {
                var row = result.GetRowSpanOfBytes(y);
                for (var x = 0; x < size.Width; x++)
                    row[x] = values[offset++] ? (byte)1 : (byte)0;
            }
            return result;
        }
        catch
        {
            result.Dispose();
            throw;
        }
    }

    extension(Mat src)
    {
        /// <summary>Normalizes an 8-bit or boolean mask into a caller-owned boolean mask, treating every nonzero value as true.</summary>
        /// <returns>A single-channel CvBool mask containing zero and one.</returns>
        public Mat ToBooleanMask() => NormalizeMask(src, DepthType.CvBool, 1);

        /// <summary>Normalizes an 8-bit or boolean mask into a caller-owned 8-bit mask suitable for contours and image codecs.</summary>
        /// <returns>A single-channel Cv8U mask containing zero and 255.</returns>
        public Mat ToByteMask() => NormalizeMask(src, DepthType.Cv8U, 255);

        /// <summary>Computes logical intersection of equally sized 8-bit or boolean masks; the inputs are preserved.</summary>
        /// <param name="other">The other single-channel 8-bit or boolean mask.</param>
        /// <returns>A caller-owned boolean mask.</returns>
        public Mat MaskAnd(Mat other) => CombineMasks(src, other, MaskCombination.And);

        /// <summary>Computes logical union of equally sized 8-bit or boolean masks; the inputs are preserved.</summary>
        /// <param name="other">The other single-channel 8-bit or boolean mask.</param>
        /// <returns>A caller-owned boolean mask.</returns>
        public Mat MaskOr(Mat other) => CombineMasks(src, other, MaskCombination.Or);

        /// <summary>Computes logical exclusive-or of equally sized 8-bit or boolean masks; the inputs are preserved.</summary>
        /// <param name="other">The other single-channel 8-bit or boolean mask.</param>
        /// <returns>A caller-owned boolean mask.</returns>
        public Mat MaskXor(Mat other) => CombineMasks(src, other, MaskCombination.Xor);

        /// <summary>Logically negates an 8-bit or boolean mask, preserving the source.</summary>
        /// <returns>A caller-owned boolean mask containing zero and one.</returns>
        public Mat MaskNot() => NormalizeMask(src, DepthType.CvBool, 1, invert: true);

        /// <summary>
        /// Finds the connected regions of nonzero pixels in an 8-bit or boolean mask and returns their statistics as
        /// managed values (no native objects to dispose).
        /// </summary>
        /// <param name="connectivity"><see cref="LineType.EightConnected"/> (diagonal neighbors join regions) or
        /// <see cref="LineType.FourConnected"/>.</param>
        /// <param name="minimumArea">Regions with fewer pixels than this are omitted.</param>
        /// <returns>The regions ordered by label, excluding the background.</returns>
        /// <exception cref="ArgumentException">The mask is not a nonempty single-channel CvBool or Cv8U mask.</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="connectivity"/> is not 4 or 8-connected, or
        /// <paramref name="minimumArea"/> is negative.</exception>
        public MaskComponent[] GetMaskComponents(
            LineType connectivity = LineType.EightConnected,
            int minimumArea = 0
        )
        {
            ValidateLogicalMask(src, nameof(src));
            if (connectivity is not (LineType.EightConnected or LineType.FourConnected))
                throw new ArgumentOutOfRangeException(nameof(connectivity), connectivity, null);
            ArgumentOutOfRangeException.ThrowIfNegative(minimumArea);

            // Byte header so CvBool and Cv8U masks (including ROI views) are labeled the same way: nonzero is foreground
            using var inputBytes = new Mat(src.Size, DepthType.Cv8U, 1, src.DataPointer, src.Step);
            using var labels = new Mat();
            using var stats = new Mat();
            using var centroids = new Mat();
            var count = CvInvoke.ConnectedComponentsWithStats(
                inputBytes,
                labels,
                stats,
                centroids,
                connectivity,
                DepthType.Cv32S,
                ConnectedComponentsAlgorithmsTypes.Default
            );
            GC.KeepAlive(src);

            // stats: count x 5 CV_32S (left, top, width, height, area); centroids: count x 2 CV_64F (x, y)
            var statValues = stats.GetReadOnlySpan<int>();
            var centroidValues = centroids.GetReadOnlySpan<double>();
            var components = new List<MaskComponent>(Math.Max(0, count - 1));
            for (var label = 1; label < count; label++)
            {
                var stat = statValues.Slice(label * 5, 5);
                var area = stat[(int)ConnectedComponentsTypes.Area];
                if (area < minimumArea)
                    continue;
                components.Add(
                    new MaskComponent(
                        label,
                        new Rectangle(
                            stat[(int)ConnectedComponentsTypes.Left],
                            stat[(int)ConnectedComponentsTypes.Top],
                            stat[(int)ConnectedComponentsTypes.Width],
                            stat[(int)ConnectedComponentsTypes.Height]
                        ),
                        area,
                        new PointF((float)centroidValues[label * 2], (float)centroidValues[label * 2 + 1])
                    )
                );
            }

            return components.ToArray();
        }

        /// <summary>
        /// Creates a boolean mask of the pixels whose color is perceptually close to <paramref name="color"/>,
        /// measured as the Euclidean distance in the Oklab color space (OpenCV 5.1+).
        /// </summary>
        /// <remarks>Oklab distances are roughly uniform to human perception: lightness is in [0, 1], so a tolerance
        /// of about 0.02 is barely noticeable, 0.05 is a small difference and 0.1 or more is a clearly different
        /// color. The alpha channel of a BGRA source is ignored.</remarks>
        /// <param name="color">The reference color, in BGR order and in the value range of the source depth
        /// (0-255 for Cv8U, 0-65535 for Cv16U, 0-1 for Cv32F).</param>
        /// <param name="tolerance">The maximum Oklab distance (inclusive) for a pixel to be selected.</param>
        /// <returns>A caller-owned single-channel CvBool mask containing zero and one.</returns>
        /// <exception cref="ArgumentException">The source is empty, not 3 (BGR) or 4 (BGRA) channels, or not Cv8U,
        /// Cv16U or Cv32F.</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="tolerance"/> is negative or not finite.
        /// </exception>
        public Mat CreateOklabDistanceMask(MCvScalar color, double tolerance)
        {
            ArgumentNullException.ThrowIfNull(src);
            if (src.IsEmpty || src.NumberOfChannels is not (3 or 4))
                throw new ArgumentException("A nonempty BGR or BGRA image is required.", nameof(src));
            var scale = src.Depth switch
            {
                DepthType.Cv8U => 1.0 / byte.MaxValue,
                DepthType.Cv16U => 1.0 / ushort.MaxValue,
                DepthType.Cv32F => 1.0,
                _ => throw new ArgumentException("The image depth must be Cv8U, Cv16U or Cv32F.", nameof(src)),
            };
            if (!double.IsFinite(tolerance))
                throw new ArgumentOutOfRangeException(nameof(tolerance), tolerance, "The tolerance must be finite.");
            ArgumentOutOfRangeException.ThrowIfNegative(tolerance);

            // Float BGR in [0, 1], so the conversion is not quantized to 8 bits
            using var bgr = new Mat();
            if (src.NumberOfChannels == 4)
            {
                CvInvoke.CvtColor(src, bgr, ColorConversion.Bgra2Bgr);
                bgr.ConvertTo(bgr, DepthType.Cv32F, scale);
            }
            else
            {
                src.ConvertTo(bgr, DepthType.Cv32F, scale);
            }

            using var lab = new Mat();
            CvInvoke.CvtColor(bgr, lab, ColorConversion.Bgr2Oklab);

            using var reference = new Mat(1, 1, DepthType.Cv32F, 3);
            reference.SetTo(new MCvScalar(color.V0 * scale, color.V1 * scale, color.V2 * scale));
            CvInvoke.CvtColor(reference, reference, ColorConversion.Bgr2Oklab);
            var referenceLab = reference.GetReadOnlySpan<float>();

            // Squared distance: sum over channels of (lab - reference)^2
            using var difference = new Mat();
            using var referenceScalar = new ScalarArray(
                new MCvScalar(referenceLab[0], referenceLab[1], referenceLab[2])
            );
            CvInvoke.Subtract(lab, referenceScalar, difference);
            CvInvoke.Multiply(difference, difference, difference);
            using var squaredDistance = new Mat();
            using var channelSum = new Mat(1, 3, DepthType.Cv32F, 1);
            channelSum.SetTo(new MCvScalar(1));
            CvInvoke.Transform(difference, squaredDistance, channelSum);

            var result = new Mat(src.Size, DepthType.CvBool, 1);
            try
            {
                // Byte header so Compare/Threshold write 0/1 into the CvBool storage without reallocating it
                using var outputBytes = new Mat(
                    result.Size,
                    DepthType.Cv8U,
                    1,
                    result.DataPointer,
                    result.Step
                );
                using var maximum = new ScalarArray(tolerance * tolerance);
                CvInvoke.Compare(squaredDistance, maximum, outputBytes, CmpType.LessEqual);
                CvInvoke.Threshold(outputBytes, outputBytes, 0, 1, ThresholdType.Binary);
                return result;
            }
            catch
            {
                result.Dispose();
                throw;
            }
        }
    }

    private enum MaskCombination
    {
        And,
        Or,
        Xor,
    }

    private static Mat CombineMasks(Mat src, Mat other, MaskCombination combination)
    {
        ValidateLogicalMask(src, nameof(src));
        ValidateLogicalMask(other, nameof(other));
        if (src.Size != other.Size)
            throw new ArgumentException("Mask dimensions must match.", nameof(other));

        var result = new Mat(src.Size, DepthType.CvBool, 1);
        try
        {
            if (combination == MaskCombination.Or)
            {
                // Raw bitwise OR is nonzero exactly when either input is nonzero.
                using var leftBytes = new Mat(
                    src.Size,
                    DepthType.Cv8U,
                    1,
                    src.DataPointer,
                    src.Step
                );
                using var rightBytes = new Mat(
                    other.Size,
                    DepthType.Cv8U,
                    1,
                    other.DataPointer,
                    other.Step
                );
                using var outputBytes = new Mat(
                    result.Size,
                    DepthType.Cv8U,
                    1,
                    result.DataPointer,
                    result.Step
                );
                CvInvoke.BitwiseOr(leftBytes, rightBytes, outputBytes);
                CvInvoke.Threshold(outputBytes, outputBytes, 0, 1, ThresholdType.Binary);
                GC.KeepAlive(src);
                GC.KeepAlive(other);
            }
            else
            {
                // AND/XOR need canonical inputs: arbitrary nonzero bytes are not logical values.
                using var left = NormalizeMask(src, DepthType.CvBool, 1);
                using var right = NormalizeMask(other, DepthType.CvBool, 1);
                switch (combination)
                {
                    case MaskCombination.And:
                        CvInvoke.BitwiseAnd(left, right, result);
                        break;
                    case MaskCombination.Xor:
                        CvInvoke.BitwiseXor(left, right, result);
                        break;
                    default:
                        throw new ArgumentOutOfRangeException(nameof(combination));
                }
            }
            return result;
        }
        catch
        {
            result.Dispose();
            throw;
        }
    }

    private static Mat NormalizeMask(Mat src, DepthType depth, byte trueValue, bool invert = false)
    {
        ValidateLogicalMask(src, nameof(src));
        var result = new Mat(src.Size, depth, 1);
        try
        {
            // Borrow byte headers so Threshold writes 0/1 into CvBool storage without changing its type.
            // The headers own no pixel memory and preserve ROI strides.
            using var inputBytes = new Mat(src.Size, DepthType.Cv8U, 1, src.DataPointer, src.Step);
            using var outputBytes = new Mat(
                result.Size,
                DepthType.Cv8U,
                1,
                result.DataPointer,
                result.Step
            );
            CvInvoke.Threshold(
                inputBytes,
                outputBytes,
                0,
                trueValue,
                invert ? ThresholdType.BinaryInv : ThresholdType.Binary
            );
            GC.KeepAlive(src);
            return result;
        }
        catch
        {
            result.Dispose();
            throw;
        }
    }

    private static void ValidateLogicalMask(Mat mask, string parameterName)
    {
        ArgumentNullException.ThrowIfNull(mask, parameterName);
        if (
            mask.IsEmpty
            || mask.NumberOfChannels != 1
            || mask.Depth is not (DepthType.CvBool or DepthType.Cv8U)
        )
            throw new ArgumentException(
                "A nonempty single-channel CvBool or Cv8U mask is required.",
                parameterName
            );
    }
}
