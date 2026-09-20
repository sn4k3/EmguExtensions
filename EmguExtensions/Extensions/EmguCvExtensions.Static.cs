/*
 *   MIT License
 *
 *   Copyright (c) 2026 Tiago Conceição
 *
 *   Permission is hereby granted, free of charge, to any person obtaining a copy
 *   of this software and associated documentation files (the "Software"), to deal
 *   in the Software without restriction, including without limitation the rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:
 *
 *   The above copyright notice and this permission notice shall be included in all
 *   copies or substantial portions of the Software.
 *
 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *   SOFTWARE.
 */

using System.Drawing;
using System.Runtime.CompilerServices;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace EmguExtensions;

public static partial class EmguCvExtensions
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int ResolveElementRange(
        int byteLength,
        int elementSize,
        int length,
        int offset,
        out int byteOffset)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(length);
        ArgumentOutOfRangeException.ThrowIfNegative(offset);

        var availableElements = byteLength / elementSize;
        if (offset > availableElements)
            throw new ArgumentOutOfRangeException(nameof(offset), offset,
                $"Offset must not exceed the available element count ({availableElements}).");

        byteOffset = offset * elementSize;
        var remainingElements = (byteLength - byteOffset) / elementSize;

        if (length == 0) return remainingElements;
        if (length > remainingElements)
            throw new ArgumentOutOfRangeException(nameof(length), length,
                $"Length must not exceed the remaining element count ({remainingElements}).");

        return length;
    }

    /// <summary>
    /// Correct openCV thickness which always results larger than specified
    /// </summary>
    /// <param name="thickness">Thickness to correct</param>
    /// <returns></returns>
    public static int CorrectThickness(int thickness)
    {
        if (thickness < 3) return thickness;
        return thickness - 1;
    }

    /// <summary>
    /// Trim the line according to the specified alignment. For None it only trims the end, for others it trims both sides.
    /// </summary>
    /// <param name="line">The line of text to trim.</param>
    /// <param name="lineAlignment">The alignment option to determine how the line should be trimmed.</param>
    /// <returns>The trimmed line of text.</returns>
    /// <exception cref="ArgumentOutOfRangeException"></exception>
    public static string PutTextLineAlignmentTrim(string line, PutTextLineAlignment lineAlignment)
    {
        ArgumentNullException.ThrowIfNull(line);
        return lineAlignment switch
        {
            PutTextLineAlignment.Default => line.TrimEnd(),
            PutTextLineAlignment.Left or PutTextLineAlignment.Center or PutTextLineAlignment.Right => line.Trim(),
            _ => throw new ArgumentOutOfRangeException(nameof(lineAlignment), lineAlignment, null)
        };
    }

    /// <summary>
    /// Get the size of a text with multiple lines, considering line gaps and alignment. It trims the text according to the alignment, for None it only trims the end, for others it trims both sides.
    /// </summary>
    /// <param name="text">The text to measure.</param>
    /// <param name="fontFace">The font face to use.</param>
    /// <param name="fontScale">The scale factor for the font.</param>
    /// <param name="thickness">The thickness of the text.</param>
    /// <param name="baseLine">The baseline of the text.</param>
    /// <param name="lineAlignment">The alignment option for the text.</param>
    /// <returns></returns>
    public static Size GetTextSizeExtended(string text, FontFace fontFace, double fontScale, int thickness,
        ref int baseLine, PutTextLineAlignment lineAlignment = default)
    {
        return GetTextSizeExtended(text, fontFace, fontScale, thickness, 0, ref baseLine, lineAlignment);
    }

    /// <summary>
    /// Get the size of a text with multiple lines, considering line gaps and alignment. It trims the text according to the alignment, for None it only trims the end, for others it trims both sides.
    /// </summary>
    /// <param name="text">The text to measure.</param>
    /// <param name="fontFace">The font face to use.</param>
    /// <param name="fontScale">The scale factor for the font.</param>
    /// <param name="thickness">The thickness of the text.</param>
    /// <param name="lineGapOffset">The offset to apply to the line gap.</param>
    /// <param name="baseLine">The baseline of the text.</param>
    /// <param name="lineAlignment">The alignment option for the text.</param>
    /// <returns>The size of the text.</returns>
    public static Size GetTextSizeExtended(string text, FontFace fontFace, double fontScale, int thickness,
        int lineGapOffset, ref int baseLine, PutTextLineAlignment lineAlignment = default)
    {
        ArgumentNullException.ThrowIfNull(text);
        text = text.TrimEnd('\n', '\r', ' ');
        var lines = text.Split(StaticObjects.LineBreakCharacters, StringSplitOptions.None);
        var firstNonEmpty = lines[0];
        for (var i = 0; i < lines.Length; i++)
        {
            if (!string.IsNullOrWhiteSpace(lines[i]))
            {
                firstNonEmpty = lines[i];
                break;
            }
        }
        var textSize = CvInvoke.GetTextSize(firstNonEmpty, fontFace, fontScale, thickness, ref baseLine);

        if (lines.Length is 0 or 1) return textSize;

        var lineGap = textSize.Height / 3 + lineGapOffset;
        var width = 0;
        var height = lines.Length * (lineGap + textSize.Height) - lineGap;

        for (var i = 0; i < lines.Length; i++)
        {
            lines[i] = PutTextLineAlignmentTrim(lines[i], lineAlignment);

            if (string.IsNullOrWhiteSpace(lines[i])) continue;
            var baseLineRef = 0;
            var lineSize = CvInvoke.GetTextSize(lines[i], fontFace, fontScale, thickness, ref baseLineRef);
            width = Math.Max(width, lineSize.Width);
        }


        return new Size(width, height);
    }

    /// <summary>
    /// Creates a new <see cref="Mat"/> and zero it
    /// </summary>
    /// <param name="size">The size of the Mat.</param>
    /// <param name="channels">The number of channels.</param>
    /// <param name="depthType">The depth type of the Mat.</param>
    /// <returns></returns>
    public static Mat InitMat(Size size, int channels = 1, DepthType depthType = DepthType.Cv8U)
    {
        return size.Width <= 0 || size.Height <= 0 ? new Mat() : Mat.Zeros(size.Height, size.Width, depthType, channels);
    }

    /// <summary>
    /// Creates a new <see cref="Mat"/> and set it to a <see cref="MCvScalar"/>
    /// </summary>
    /// <param name="size">The size of the Mat.</param>
    /// <param name="color">The color to set the Mat to.</param>
    /// <param name="channels">The number of channels.</param>
    /// <param name="depthType">The depth type of the Mat.</param>
    /// <param name="mask">An optional mask to apply when setting the color.</param>
    /// <returns>The initialized Mat.</returns>
    public static Mat InitMat(Size size, MCvScalar color, int channels = 1, DepthType depthType = DepthType.Cv8U,
        IInputArray? mask = null)
    {
        if (size.Width <= 0 || size.Height <= 0) return new Mat();
        var mat = new Mat(size, depthType, channels);
        try
        {
            mat.SetTo(color, mask);
        }
        catch
        {
            mat.Dispose();
            throw;
        }

        return mat;
    }

    /// <summary>
    /// Creates an array of new <see cref="Mat"/> and zero it
    /// </summary>
    /// <param name="count">The number of Mats to create.</param>
    /// <param name="size">The size of each Mat.</param>
    /// <param name="channels">The number of channels for each Mat.</param>
    /// <param name="depthType">The depth type for each Mat.</param>
    /// <returns>An array of initialized Mats.</returns>
    public static Mat[] InitMats(int count, Size size, int channels = 1, DepthType depthType = DepthType.Cv8U)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(count);
        var mats = new Mat[count];
        var initializedCount = 0;
        try
        {
            for (; initializedCount < count; initializedCount++)
            {
                mats[initializedCount] = InitMat(size, channels, depthType);
            }
        }
        catch
        {
            for (var i = 0; i < initializedCount; i++)
            {
                mats[i].Dispose();
            }

            throw;
        }

        return mats;
    }

    /// <summary>
    /// Creates a new <see cref="Mat"/> from a <see cref="ReadOnlySpan{T}"/> of bytes. The Mat will have 1 row and the length of the span as columns, with a depth type of Cv8U and 1 channel.
    /// </summary>
    /// <param name="buffer">The span of bytes to create the Mat from.</param>
    /// <returns>A new Mat initialized with the data from the span.</returns>
    /// <exception cref="ArgumentException">Thrown if the span is empty.</exception>
    /// <remarks>
    /// WARNING: This method returns a <see cref="Mat"/> pointing directly to the memory of <paramref name="buffer"/>.
    /// If the buffer is backed by managed memory (e.g., an unpinned byte array), it must remain fixed/pinned in memory
    /// for the lifetime of the returned <see cref="Mat"/> to prevent memory corruption.
    /// All manipulations on the returned <see cref="Mat"/> will reflect in the original buffer.
    /// </remarks>
    public static Mat CreateVector(ReadOnlySpan<byte> buffer)
    {
        if (buffer.IsEmpty)
            throw new ArgumentException("The buffer must not be empty.", nameof(buffer));

        unsafe
        {
            fixed (byte* ptr = buffer)
            {
                return new Mat(1, buffer.Length, DepthType.Cv8U, 1, (IntPtr)ptr, buffer.Length);
            }
        }
    }

    #region Kernel methods

    /// <summary>
    /// Reduces iterations to 1 and generate a kernel to match the iterations effect
    /// </summary>
    /// <param name="iterations"></param>
    /// <param name="elementShape"></param>
    /// <returns></returns>
    public static Mat CreateDynamicKernel(ref int iterations, MorphShapes elementShape = MorphShapes.Ellipse)
    {
        var size = Math.Max(iterations, 1) * 2 + 1;
        iterations = 1;
        return CvInvoke.GetStructuringElement(elementShape, new Size(size, size), AnchorCenter);
    }

    #endregion
}
