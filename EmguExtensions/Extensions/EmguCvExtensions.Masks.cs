using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;

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
