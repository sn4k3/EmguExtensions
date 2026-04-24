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

using CommunityToolkit.HighPerformance;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System.Drawing;
using System.IO.Compression;
using System.IO.Hashing;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using Point = System.Drawing.Point;
using Size = System.Drawing.Size;

namespace EmguExtensions;

public static partial class EmguExtensions
{
    extension(DepthType src)
    {
        /// <summary>
        /// Gets the byte count for a single channel of the specified depth type. This value represents the number of bytes required to store one pixel value for a single channel of the given depth.
        /// </summary>
        public byte ByteCount =>
            src switch
            {
                DepthType.Default => 1,
                DepthType.Cv8U => 1,
                DepthType.Cv8S => 1,
                DepthType.Cv16U => 2,
                DepthType.Cv16S => 2,
                DepthType.Cv32S => 4,
                DepthType.Cv32F => 4,
                DepthType.Cv64F => 8,
                _ => throw new ArgumentOutOfRangeException()
            };
    }

    extension(IInputArray src)
    {
        /// <summary>
        /// Gets a value indicating whether all pixels in the matrix are zero.
        /// </summary>
        /// <remarks>This value is not cached and recalculated each time it is accessed.</remarks>
        public bool IsAllZero => !CvInvoke.HasNonZero(src);

        /// <summary>
        /// Gets a value indicating whether there is at least one non-zero pixel in the matrix.
        /// </summary>
        /// <remarks>This value is not cached and recalculated each time it is accessed.</remarks>
        public bool HasNonZero => CvInvoke.HasNonZero(src);

        /// <summary>
        /// Gets the number of non-zero pixels in the matrix.
        /// </summary>
        /// <returns>The count of non-zero elements.</returns>
        /// <remarks>This value is not cached and recalculated each time it is accessed.</remarks>
        public int CountNonZero => CvInvoke.CountNonZero(src);
    }

    extension(IInputOutputArray src)
    {
        /// <summary>
        /// Retrieves contours from the binary image as a contour tree. The pointer firstContour is filled by the function. It is provided as a convenient way to obtain the hierarchy value as int[,].
        /// The function modifies the source image content
        /// </summary>
        /// <param name="mode">Retrieval mode</param>
        /// <param name="method">Approximation method (for all the modes, except CV_RETR_RUNS, which uses built-in approximation). </param>
        /// <param name="offset">Offset, by which every contour point is shifted. This is useful if the contours are extracted from the image ROI and then they should be analyzed in the whole image context</param>
        /// <returns>The contour hierarchy</returns>
        public VectorOfVectorOfPoint FindContours(RetrType mode = RetrType.List, ChainApproxMethod method = ChainApproxMethod.ChainApproxSimple, Point offset = default)
        {
            using var hierarchy = new Mat();
            var contours = new VectorOfVectorOfPoint();
            CvInvoke.FindContours(src, contours, hierarchy, mode, method, offset);
            return contours;
        }

        /// <summary>
        /// Retrieves contours from the binary image as a contour tree. The pointer firstContour is filled by the function. It is provided as a convenient way to obtain the hierarchy value as int[,].
        /// The function modifies the source image content
        /// </summary>
        /// <param name="hierarchy">The contour hierarchy</param>
        /// <param name="mode">Retrieval mode</param>
        /// <param name="method">Approximation method (for all the modes, except CV_RETR_RUNS, which uses built-in approximation). </param>
        /// <param name="offset">Offset, by which every contour point is shifted. This is useful if the contours are extracted from the image ROI and then they should be analyzed in the whole image context</param>
        /// <returns>Detected contours. Each contour is stored as a vector of points.</returns>
        public VectorOfVectorOfPoint FindContours(out int[,] hierarchy, RetrType mode, ChainApproxMethod method = ChainApproxMethod.ChainApproxSimple, Point offset = default)
        {
            var contours = new VectorOfVectorOfPoint();
            using var hierarchyMat = new Mat();

            CvInvoke.FindContours(src, contours, hierarchyMat, mode, method, offset);

            hierarchy = new int[hierarchyMat.Cols, 4];
            if (contours.Size == 0) return contours;
            using var gcHandle = new GCSafeHandle(hierarchy);
            using var mat2 = new Mat(hierarchyMat.Rows, hierarchyMat.Cols, hierarchyMat.Depth, 4, gcHandle.DangerousGetHandle(), hierarchyMat.Step);
            hierarchyMat.CopyTo(mat2);
            return contours;
        }


        /// <summary>
        /// Encodes the source image as a PNG and returns the resulting byte array.
        /// </summary>
        /// <returns>A byte array containing the PNG-encoded image data.</returns>
        public byte[] GetPngBytes()
        {
            return CvInvoke.Imencode(".png", src);
        }

        /// <summary>
        /// Encodes the source image as a PNG and returns the resulting byte array.
        /// </summary>
        /// <param name="pngCompressionLevel">The compression level to use for PNG encoding. Valid values typically range from 0 (no compression) to 9
        /// (maximum compression). The default is 3.</param>
        /// <returns>A byte array containing the PNG-encoded image data.</returns>
        public byte[] GetPngBytes(int pngCompressionLevel)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(pngCompressionLevel);
            ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(pngCompressionLevel, 10);
            return CvInvoke.Imencode(".png", src, new KeyValuePair<ImwriteFlags, int>(ImwriteFlags.PngCompression, pngCompressionLevel));
        }

        /// <summary>
        /// Encodes the source image as a PNG and returns the resulting byte array.
        /// </summary>
        /// <param name="compressionLevel">The compression level to use for PNG encoding. Valid values typically range from 0 (no compression) to 9
        /// (maximum compression). The default is <see cref="CompressionLevel.Optimal"/>=3.</param>
        /// <returns>A byte array containing the PNG-encoded image data.</returns>
        public byte[] GetPngBytes(CompressionLevel compressionLevel)
        {
            var pngCompressionLevel = compressionLevel switch
            {
                CompressionLevel.NoCompression => 0,
                CompressionLevel.Fastest => 1,
                CompressionLevel.Optimal => 3,
                CompressionLevel.SmallestSize => 9,
                _ => throw new ArgumentOutOfRangeException(nameof(compressionLevel), compressionLevel, null)
            };
            return src.GetPngBytes(pngCompressionLevel);
        }

        /// <summary>
        ///Draws the specified text string on the image at the given location, using the specified font, scale, color,
        /// and formatting options.
        /// </summary>
        /// <remarks>This method supports advanced text rendering options, including line alignment and
        /// origin control. The appearance of the text may vary depending on the font face and line type
        /// selected.</remarks>
        /// <param name="text">The text string to be drawn on the image.</param>
        /// <param name="org">The bottom-left corner of the first character of the text, specified as a point in image coordinates.</param>
        /// <param name="fontFace">The font type to be used for rendering the text.</param>
        /// <param name="fontScale">The scale factor that determines the size of the text. Must be positive.</param>
        /// <param name="color">The color of the text, specified as a scalar value.</param>
        /// <param name="thickness">The thickness of the text strokes, in pixels. Must be greater than or equal to 1.</param>
        /// <param name="lineType">The type of the line used to draw the text, which affects the appearance of the text edges.</param>
        /// <param name="bottomLeftOrigin">If set to <see langword="true"/>, the origin is at the bottom-left corner. Otherwise, the origin is at the
        /// top-left.</param>
        /// <param name="lineAlignment">Specifies the alignment of multi-line text relative to the origin point.</param>
        public void PutTextExtended(string text, Point org, FontFace fontFace, double fontScale,
            MCvScalar color, int thickness = 1, LineType lineType = LineType.EightConnected,
            bool bottomLeftOrigin = false, PutTextLineAlignment lineAlignment = default)
            => src.PutTextExtended(text, org, fontFace, fontScale, color, thickness, 0, lineType, bottomLeftOrigin, lineAlignment);

        /// <summary>
        /// Extended OpenCV PutText to accepting line breaks and line alignment
        /// </summary>
        public void PutTextExtended(string text, Point org, FontFace fontFace, double fontScale,
            MCvScalar color, int thickness = 1, int lineGapOffset = 0, LineType lineType = LineType.EightConnected, bool bottomLeftOrigin = false, PutTextLineAlignment lineAlignment = default)
        {
            text = text.TrimEnd('\n', '\r', ' ');
            var lines = text.Split(StaticObjects.LineBreakCharacters, StringSplitOptions.None);

            switch (lines.Length)
            {
                case 0:
                    return;
                case 1:
                    CvInvoke.PutText(src, text, org, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin);
                    return;
            }

            // Get height of a single line in pixels (all lines share the same height)
            int baseLine = 0;
            var firstNonEmptyLine = Array.Find(lines, l => !string.IsNullOrWhiteSpace(l)) ?? lines[0];
            var textSize = CvInvoke.GetTextSize(firstNonEmptyLine, fontFace, fontScale, thickness, ref baseLine);
            var lineGap = textSize.Height / 3 + lineGapOffset;
            var linesSize = new Size[lines.Length];
            int width = 0;

            // Sanitize lines
            for (var i = 0; i < lines.Length; i++)
            {
                lines[i] = PutTextLineAlignmentTrim(lines[i], lineAlignment);
            }

            // If line needs alignment, calculate the size for each line
            if (lineAlignment is not PutTextLineAlignment.Left and not PutTextLineAlignment.None)
            {
                for (var i = 0; i < lines.Length; i++)
                {
                    if (string.IsNullOrWhiteSpace(lines[i])) continue;
                    int baseLineRef = 0;
                    linesSize[i] = CvInvoke.GetTextSize(lines[i], fontFace, fontScale, thickness, ref baseLineRef);
                    width = Math.Max(width, linesSize[i].Width);
                }
            }

            for (var i = 0; i < lines.Length; i++)
            {
                if (string.IsNullOrWhiteSpace(lines[i])) continue;

                int x = lineAlignment switch
                {
                    PutTextLineAlignment.None or PutTextLineAlignment.Left => org.X,
                    PutTextLineAlignment.Center => org.X + (width - linesSize[i].Width) / 2,
                    PutTextLineAlignment.Right => org.X + width - linesSize[i].Width,
                    _ => throw new ArgumentOutOfRangeException(nameof(lineAlignment), lineAlignment, null)
                };

                // Find total size of text block before this line
                var lineYAdjustment = i * (lineGap + textSize.Height);
                // Move text down from original line based on line number
                int lineY = !bottomLeftOrigin ? org.Y + lineYAdjustment : org.Y - lineYAdjustment;
                CvInvoke.PutText(src, lines[i], new Point(x, lineY), fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin);
            }
        }
    }

    /// <param name="src">Source <see cref="Mat"/></param>
    extension(Mat src)
    {
        #region Properties

        /// <summary>
        /// Gets the real step size in bytes for the current matrix width.
        /// Unlike <see cref="Mat.Step"/>, which may reflect the original matrix step even for ROIs,
        /// this property always returns the correct step based on the current width.
        /// </summary>
        public int RealStep => src.GetByteCount(src.Width);

        /// <summary>
        /// Gets the total length of the matrix data in bytes, represented as a 32-bit integer.
        /// </summary>
        /// <remarks>This property is intended for use with matrices whose total data size does not exceed
        /// the maximum value of a 32-bit integer. For larger matrices, use a property that returns a 64-bit length to
        /// avoid overflow.</remarks>
        public int LengthInt32 => src.GetByteCount(src.Total.ToInt32());

        /// <summary>
        /// Gets the total length of the matrix data, in bytes, as a 64-bit integer.
        /// </summary>
        /// <remarks>Use this property when working with large matrices that may exceed the range of a
        /// 32-bit integer. The value represents the total number of bytes required to store all elements of the matrix,
        /// including all channels.</remarks>
        public long LengthInt64 => src.GetByteCount(src.Total.ToInt64());

        /// <summary>
        /// Calculates the center point of the matrix based on its width and height. The center point is determined by dividing the width and height by 2, resulting in a Point that represents the coordinates of the center pixel of the matrix.
        /// </summary>
        public Point CenterPoint => new(src.Width / 2, src.Height / 2);

        #endregion

        #region Initializer methods
        /// <summary>
        /// Creates a new <see cref="Mat"/> with same size and type of the source
        /// </summary>
        /// <returns></returns>
        public Mat New() => new(src.Size, src.Depth, src.NumberOfChannels);

        /// <summary>
        /// Creates a <see cref="Mat"/> with same size and type of the source and set it to a color
        /// </summary>
        /// <param name="color"></param>
        /// <param name="mask"></param>
        /// <returns></returns>
        public Mat NewSetTo(MCvScalar color, IInputArray? mask = null) => InitMat(src.Size, color, src.NumberOfChannels, src.Depth, mask);

        /// <summary>
        /// Creates a new blanked (All zeros) <see cref="Mat"/> with same size and type of the source
        /// </summary>
        /// <returns>Blanked <see cref="Mat"/></returns>
        public Mat NewZeros() => InitMat(src.Size, src.NumberOfChannels, src.Depth);
        #endregion

        #region Copy methods

        /// <summary>
        /// Copies the contents of the matrix to the specified unmanaged memory location.
        /// </summary>
        /// <remarks>This method performs a direct memory copy of the matrix data to unmanaged memory. The
        /// caller is responsible for ensuring that the destination pointer is valid and that the memory region is
        /// appropriately sized. No bounds checking is performed on the destination buffer.</remarks>
        /// <param name="destination">A pointer to the destination memory where the matrix data will be copied. The destination buffer must be
        /// large enough to hold the entire contents of the matrix.</param>
        public void CopyTo(IntPtr destination)
        {
            if (destination == IntPtr.Zero)
                throw new ArgumentNullException(nameof(destination));
            unsafe
            {
                if (src.IsContinuous)
                {
                    var totalBytes = src.LengthInt64;
                    Buffer.MemoryCopy(src.DataPointer.ToPointer(), destination.ToPointer(), totalBytes, totalBytes);
                }
                else
                {
                    var srcSpan = src.GetSpan2D<byte>();
                    var dstSpan = new Span<byte>(destination.ToPointer(), (int)srcSpan.Length);
                    srcSpan.CopyTo(dstSpan);


                    /*// Alternative:
                    // Row by row copy
                    for (int y = 0; y < src.Height; y++)
                    {
                        Buffer.MemoryCopy(sourcePtr + y * sourceStep, destPtr + y * destStep, destStep, bytesToCopyPerRow);
                    }
                    */
                }
            }
        }

        /// <summary>
        /// Copies the contents of the matrix to the specified stream.
        /// </summary>
        /// <param name="stream"></param>
        public void CopyTo(Stream stream)
        {
            if (src.IsContinuous)
            {
                stream.Write(src.GetSpan<byte>());
            }
            else
            {
                // Non-continuous Mat: write row by row
                var span2D = src.GetSpan2D<byte>();
                for (var row = 0; row < span2D.Height; row++)
                {
                    stream.Write(span2D.GetRowSpan(row));
                }
            }
        }

        /// <summary>
        /// Copies the contents of the current matrix onto a destination matrix at the specified offset,
        /// optionally using a mask to control which pixels are copied.
        /// </summary>
        /// <remarks>The source matrix is clipped to fit within the destination bounds when placed at the
        /// given offset. If a mask is provided, only pixels where the mask is non-zero are copied. The mask must be
        /// single-channel 8-bit and have the same size as the source matrix (or the clipped region).</remarks>
        /// <param name="destination">The destination matrix onto which the source will be copied. Must not be null.</param>
        /// <param name="offset">The top-left position in the destination matrix where the source will be placed.</param>
        /// <param name="mask">An optional single-channel 8-bit mask controlling which pixels are copied. If null, all pixels are
        /// copied.</param>
        public void CopyTo(Mat destination, Point offset, Mat? mask = null)
        {
            // Calculate the overlapping region between source and destination
            int srcX = Math.Max(0, -offset.X);
            int srcY = Math.Max(0, -offset.Y);
            int dstX = Math.Max(0, offset.X);
            int dstY = Math.Max(0, offset.Y);

            int width = Math.Min(src.Width - srcX, destination.Width - dstX);
            int height = Math.Min(src.Height - srcY, destination.Height - dstY);

            if (width <= 0 || height <= 0) return;

            var srcRoi = new Rectangle(srcX, srcY, width, height);
            var dstRoi = new Rectangle(dstX, dstY, width, height);

            using var srcRegion = new Mat(src, srcRoi);
            using var dstRegion = new Mat(destination, dstRoi);

            if (mask is not null)
            {
                using var maskRegion = new Mat(mask, srcRoi);
                srcRegion.CopyTo(dstRegion, maskRegion);
            }
            else
            {
                srcRegion.CopyTo(dstRegion);
            }
        }

        /// <summary>
        /// Copies a center region of the specified size from the source matrix to the center of the destination matrix.
        /// </summary>
        /// <param name="size">The size of the region to copy from the center of the source matrix.</param>
        /// <param name="dst">The destination matrix to paste the region into.</param>
        public void CopyCenterToCenter(Size size, Mat dst)
        {
            using var srcRoi = src.RoiFromCenter(size);
            srcRoi.CopyToCenter(dst);
        }

        /// <summary>
        /// Copies a specified region from the source matrix to the center of the destination matrix.
        /// </summary>
        /// <param name="region">The region rectangle to copy from the source matrix.</param>
        /// <param name="dst">The destination matrix to paste the region into.</param>
        public void CopyRegionToCenter(Rectangle region, Mat dst)
        {
            using var srcRoi = src.Roi(region);
            srcRoi.CopyToCenter(dst);
        }

        /// <summary>
        /// Copies the source matrix to the center of the destination matrix.
        /// </summary>
        /// <param name="dst">The destination matrix to paste the source into.</param>
        public void CopyToCenter(Mat dst)
        {
            if (src.Size == dst.Size)
            {
                src.CopyTo(dst);
                return;
            }

            // Compute offsets per-axis independently so mixed cases (dst wider but shorter, etc.) are handled correctly
            int dstX = Math.Max(0, (dst.Width - src.Width) / 2);
            int dstY = Math.Max(0, (dst.Height - src.Height) / 2);
            int srcX = Math.Max(0, (src.Width - dst.Width) / 2);
            int srcY = Math.Max(0, (src.Height - dst.Height) / 2);
            int copyWidth = Math.Min(src.Width, dst.Width);
            int copyHeight = Math.Min(src.Height, dst.Height);

            if (copyWidth <= 0 || copyHeight <= 0) return;

            using var srcRoi = src.Roi(new Rectangle(srcX, srcY, copyWidth, copyHeight));
            using var dstRoi = dst.Roi(new Rectangle(dstX, dstY, copyWidth, copyHeight));
            srcRoi.CopyTo(dstRoi);
        }

        #endregion

        #region Memory accessors

        /// <summary>
        /// Gets the byte pointer to the underlying data of this matrix.
        /// </summary>
        public unsafe byte* BytePointer
        {
            get
            {
                if (src.IsEmpty || src.DataPointer == IntPtr.Zero)
                    throw new InvalidOperationException("Cannot access BytePointer of an empty or uninitialized Mat.");
                return (byte*)src.DataPointer.ToPointer();
            }
        }

        /// <summary>
        /// Gets the matrix data as an <see cref="UnmanagedMemoryStream"/> with the specified access mode.
        /// </summary>
        /// <param name="accessMode">The access mode for the stream (read, write, or read-write).</param>
        /// <returns>An <see cref="UnmanagedMemoryStream"/> wrapping the matrix data.</returns>
        public UnmanagedMemoryStream GetUnmanagedMemoryStream(FileAccess accessMode)
        {
            if (src.IsEmpty || src.DataPointer == IntPtr.Zero)
                throw new InvalidOperationException("Cannot create an UnmanagedMemoryStream from an empty or uninitialized Mat.");
            var length = src.LengthInt32;
            unsafe
            {
                return new UnmanagedMemoryStream(src.BytePointer, length, length, accessMode);
            }
        }

        /// <summary>
        /// Gets a span of the matrix data for manipulation or reading.
        /// The length parameter specifies the number of elements in the span, and the offset
        /// allows skipping a certain number of elements from the start of the data.
        /// This method is only applicable for continuous matrices, where all pixel data is
        /// stored in a single contiguous block of memory. If the matrix is not continuous, an
        /// exception is thrown, and users should use GetSpan2D instead to access the data in a
        /// row-wise manner. The type parameter T represents the type of data stored in the
        /// matrix (e.g., byte, float), and it must be a value type (struct). The method ensures
        /// that the requested length and offset do not exceed the bounds of the matrix data.
        /// </summary>
        /// <typeparam name="T">The type of data stored in the matrix (e.g., byte, float).</typeparam>
        /// <param name="length">The number of elements in the span.</param>
        /// <param name="offset">The number of elements to skip from the start of the data.</param>
        /// <returns>A span representing the matrix data.</returns>
        /// <exception cref="NotSupportedException">Thrown if the matrix is not continuous.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown if the offset or length is out of range.</exception>
        public Span<T> GetSpan<T>(int length, int offset) where T : struct
        {
            ArgumentOutOfRangeException.ThrowIfNegative(offset);

            if (!src.IsContinuous)
                throw new NotSupportedException("To create a Span, the Mat's memory must be continuous. This Mat does not use continuous memory. Use Span2D instead.");

            var sizeOfT = Unsafe.SizeOf<T>();
            offset *= sizeOfT;
            var maxLength = (src.LengthInt32 - offset) / sizeOfT;

            if (maxLength < 0)
                throw new ArgumentOutOfRangeException(nameof(offset), offset, "Offset value overflow this Mat size.");

            if (length <= 0)
            {
                length = maxLength;
            }
            else if (length > maxLength)
            {
                throw new ArgumentOutOfRangeException(nameof(length), length, $"The maximum size allowed for this Mat with an offset of {offset} is {maxLength}.");
            }

            unsafe
            {
                return new(IntPtr.Add(src.DataPointer, offset).ToPointer(), length);
            }
        }

        /// <summary>
        /// Gets a span of the matrix data for reading.
        /// The length parameter specifies the number of elements in the span, and the offset
        /// allows skipping a certain number of elements from the start of the data.
        /// This method is only applicable for continuous matrices, where all pixel data is
        /// stored in a single contiguous block of memory. If the matrix is not continuous, an
        /// exception is thrown, and users should use GetSpan2D instead to access the data in a
        /// row-wise manner. The type parameter T represents the type of data stored in the
        /// matrix (e.g., byte, float), and it must be a value type (struct). The method ensures
        /// that the requested length and offset do not exceed the bounds of the matrix data.
        /// </summary>
        /// <typeparam name="T">The type of data stored in the matrix (e.g., byte, float).</typeparam>
        /// <param name="length">The number of elements in the span.</param>
        /// <param name="offset">The number of elements to skip from the start of the data.</param>
        /// <returns>A span representing the matrix data.</returns>
        /// <exception cref="NotSupportedException">Thrown if the matrix is not continuous.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown if the offset or length is out of range.</exception>
        public ReadOnlySpan<T> GetReadOnlySpan<T>(int length = 0, int offset = 0) where T : struct
        {
            ArgumentOutOfRangeException.ThrowIfNegative(offset);

            if (!src.IsContinuous)
                throw new NotSupportedException("To create a Span, the Mat's memory must be continuous. This Mat does not use continuous memory. Use Span2D instead.");

            var sizeOfT = Unsafe.SizeOf<T>();
            offset *= sizeOfT;
            var maxLength = (src.LengthInt32 - offset) / sizeOfT;

            if (maxLength < 0)
                throw new ArgumentOutOfRangeException(nameof(offset), offset, "Offset value overflow this Mat size.");

            if (length <= 0)
            {
                length = maxLength;
            }
            else if (length > maxLength)
            {
                throw new ArgumentOutOfRangeException(nameof(length), length, $"The maximum size allowed for this Mat with an offset of {offset} is {maxLength}.");
            }

            unsafe
            {
                return new(IntPtr.Add(src.DataPointer, offset).ToPointer(), length);
            }
        }

        /// <summary>
        /// Gets a 2D span of the matrix data for manipulation or reading, allowing access to
        /// the data in a row-wise manner. This method is suitable for non-continuous matrices,
        /// where pixel data  may not be stored in a single contiguous block of memory.
        /// The type parameter T represents the type of data stored in the matrix
        /// (e.g., byte, float), and it must be a value type (struct).
        /// The method calculates the appropriate step size for navigating through the rows
        /// of the matrix based on its memory layout, ensuring that users can access each row
        /// correctly regardless of whether the matrix is continuous or not.
        /// </summary>
        /// <typeparam name="T">The type of data stored in the matrix (e.g., byte, float).</typeparam>
        /// <returns>A 2D span representing the matrix data.</returns>
        public Span2D<T> GetSpan2D<T>() where T : struct
        {
            var sizeOfT = Unsafe.SizeOf<T>();
            var step = src.RealStep / sizeOfT;
            unsafe
            {
                if (src.IsContinuous) return new(src.DataPointer.ToPointer(), src.Height, step, 0);
                return new(src.DataPointer.ToPointer(), src.Height, step, src.Step / sizeOfT - step);
            }
        }

        /// <summary>
        /// Gets a 2D span of the matrix data for reading, allowing access to
        /// the data in a row-wise manner. This method is suitable for non-continuous matrices,
        /// where pixel data  may not be stored in a single contiguous block of memory.
        /// The type parameter T represents the type of data stored in the matrix
        /// (e.g., byte, float), and it must be a value type (struct).
        /// The method calculates the appropriate step size for navigating through the rows
        /// of the matrix based on its memory layout, ensuring that users can access each row
        /// correctly regardless of whether the matrix is continuous or not.
        /// </summary>
        /// <typeparam name="T">The type of data stored in the matrix (e.g., byte, float).</typeparam>
        /// <returns>A 2D span representing the matrix data.</returns>
        public ReadOnlySpan2D<T> GetReadOnlySpan2D<T>() where T : struct
        {
            var sizeOfT = Unsafe.SizeOf<T>();
            var step = src.RealStep / sizeOfT;

            unsafe
            {
                if (src.IsContinuous) return new(src.DataPointer.ToPointer(), src.Height, step, 0);
                return new(src.DataPointer.ToPointer(), src.Height, step, src.Step / sizeOfT - step);
            }
        }

        /// <summary>
        /// Gets a 2D span of the matrix data within a specified region of interest (ROI),
        /// allowing access to the data in a row-wise manner.
        /// The ROI coordinates are in pixel units. The returned span width accounts for
        /// the number of channels and element size of the matrix.
        /// </summary>
        /// <typeparam name="T">The type of data stored in the matrix (e.g., byte, float).</typeparam>
        /// <param name="roi">The region of interest rectangle in pixel coordinates.</param>
        /// <returns>A 2D span representing the matrix data within the ROI.</returns>
        /// <exception cref="ArgumentOutOfRangeException">
        /// Thrown when the ROI extends beyond the matrix boundaries.
        /// </exception>
        public Span2D<T> GetSpan2D<T>(Rectangle roi) where T : struct
        {
            if (roi.IsEmpty) return Span2D<T>.Empty;
            if (roi.X < 0) throw new ArgumentOutOfRangeException(nameof(roi), $"ROI X ({roi.X}) must be non-negative.");
            if (roi.Y < 0) throw new ArgumentOutOfRangeException(nameof(roi), $"ROI Y ({roi.Y}) must be non-negative.");
            if (roi.Right > src.Width) throw new ArgumentOutOfRangeException(nameof(roi), $"ROI right edge ({roi.Right}) exceeds matrix width ({src.Width}).");
            if (roi.Bottom > src.Height) throw new ArgumentOutOfRangeException(nameof(roi), $"ROI bottom edge ({roi.Bottom}) exceeds matrix height ({src.Height}).");

            var sizeOfT = Unsafe.SizeOf<T>();
            var roiWidth = src.GetByteCount(roi.Width) / sizeOfT;
            var pitch = (src.Step / sizeOfT) - roiWidth;

            unsafe
            {
                var ptr = IntPtr.Add(src.DataPointer, (roi.Y * src.Step + src.GetByteCount(roi.X))).ToPointer();
                return new(ptr, roi.Height, roiWidth, pitch);
            }
        }

        /// <summary>
        /// Gets a 2D span of the matrix data within a specified region of interest (ROI),
        /// allowing access to the data in a row-wise manner.
        /// The ROI coordinates are in pixel units. The returned span width accounts for
        /// the number of channels and element size of the matrix.
        /// </summary>
        /// <typeparam name="T">The type of data stored in the matrix (e.g., byte, float).</typeparam>
        /// <param name="roi">The region of interest rectangle in pixel coordinates.</param>
        /// <returns>A 2D span representing the matrix data within the ROI.</returns>
        /// <exception cref="ArgumentOutOfRangeException">
        /// Thrown when the ROI extends beyond the matrix boundaries.
        /// </exception>
        public ReadOnlySpan2D<T> GetReadOnlySpan2D<T>(Rectangle roi) where T : struct
        {
            if (roi.IsEmpty) return ReadOnlySpan2D<T>.Empty;
            if (roi.X < 0) throw new ArgumentOutOfRangeException(nameof(roi), $"ROI X ({roi.X}) must be non-negative.");
            if (roi.Y < 0) throw new ArgumentOutOfRangeException(nameof(roi), $"ROI Y ({roi.Y}) must be non-negative.");
            if (roi.Right > src.Width) throw new ArgumentOutOfRangeException(nameof(roi), $"ROI right edge ({roi.Right}) exceeds matrix width ({src.Width}).");
            if (roi.Bottom > src.Height) throw new ArgumentOutOfRangeException(nameof(roi), $"ROI bottom edge ({roi.Bottom}) exceeds matrix height ({src.Height}).");

            var sizeOfT = Unsafe.SizeOf<T>();
            var roiWidth = src.GetByteCount(roi.Width) / sizeOfT;
            var pitch = (src.Step / sizeOfT) - roiWidth;

            unsafe
            {
                var ptr = IntPtr.Add(src.DataPointer, (roi.Y * src.Step + src.GetByteCount(roi.X))).ToPointer();
                return new(ptr, roi.Height, roiWidth, pitch);
            }
        }

        /// <summary>
        /// Gets a row span of the matrix data for manipulation, allowing access to a specific row of the matrix.
        /// </summary>
        /// <typeparam name="T">The type of data stored in the matrix (e.g., byte, float).</typeparam>
        /// <param name="y">The row index.</param>
        /// <param name="length">The number of elements in the row span.</param>
        /// <param name="offset">The offset within the row.</param>
        /// <returns>A span representing the specified row of the matrix.</returns>
        /// <exception cref="ArgumentOutOfRangeException"></exception>
        public Span<T> GetRowSpan<T>(int y, int length = 0, int offset = 0) where T : struct
        {
            ArgumentOutOfRangeException.ThrowIfNegative(y);
            ArgumentOutOfRangeException.ThrowIfNegative(offset);

            if (y >= src.Height)
                throw new ArgumentOutOfRangeException(nameof(y), y, $"Row index must be less than the matrix height ({src.Height}).");

            var sizeOfT = Unsafe.SizeOf<T>();

            offset *= sizeOfT;
            var maxLength = (src.RealStep - offset) / sizeOfT;

            if (maxLength < 0)
                throw new ArgumentOutOfRangeException(nameof(offset), offset, "Offset value overflow this Mat row size.");

            if (length <= 0)
            {
                length = maxLength;
            }
            else if (length > maxLength)
            {
                throw new ArgumentOutOfRangeException(nameof(length), length, $"The maximum size allowed for this Mat row with an offset of {offset} is {maxLength}.");
            }

            unsafe
            {
                return new(IntPtr.Add(src.DataPointer, (y * src.Step + offset)).ToPointer(), length);
            }
        }

        /// <summary>
        /// Gets a read-only row span of the matrix data, allowing access to a specific row of the matrix.
        /// </summary>
        /// <typeparam name="T">The type of data stored in the matrix (e.g., byte, float).</typeparam>
        /// <param name="y">The row index.</param>
        /// <param name="length">The number of elements in the row span.</param>
        /// <param name="offset">The offset within the row.</param>
        /// <returns>A read-only span representing the specified row of the matrix.</returns>
        public ReadOnlySpan<T> GetReadOnlyRowSpan<T>(int y, int length = 0, int offset = 0) where T : struct
        {
            return src.GetRowSpan<T>(y, length, offset);
        }
        #endregion

        #region Memory fill

        /// <summary>
        /// Fills a span of the matrix data with a specified value, starting from a given position
        /// and extending for a specified length. The method includes an optimization to skip
        /// filling when the provided value is below a certain threshold, which is particularly
        /// useful for performance when dealing with low-intensity colors (e.g., near black).
        /// If the value is below the threshold, the method simply advances the start position by the
        /// length without modifying the data, effectively leaving that portion of the matrix unchanged.
        /// This can save processing time when filling large areas with values that would
        /// not significantly alter the visual output.
        /// </summary>
        /// <typeparam name="T">The type of data stored in the matrix (e.g., byte, float).</typeparam>
        /// <param name="startPosition">The starting position within the matrix span.</param>
        /// <param name="length">The number of elements to fill.</param>
        /// <param name="value">The value to fill the span with.</param>
        /// <param name="valueMinThreshold">The minimum threshold value for filling. If the value is below this threshold, the fill operation is skipped.</param>
        public void FillSpan<T>(ref int startPosition, int length, T value, T valueMinThreshold) where T : struct, IComparisonOperators<T, T, bool>
        {
            if (length <= 0) return;
            if (value < valueMinThreshold) // Ignore threshold (mostly if blacks), spare cycles
            {
                startPosition += length;
                return;
            }

            src.GetSpan<T>(length, startPosition).Fill(value);
            startPosition += length;
        }

        /// <summary>
        /// Fills a span of the matrix data with a specified value, starting from a given position
        /// and extending for a specified length. The method includes an optimization to skip
        /// filling when the provided value is below a certain threshold, which is particularly
        /// useful for performance when dealing with low-intensity colors (e.g., near black).
        /// If the value is below the threshold, the method simply advances the start position by the
        /// length without modifying the data, effectively leaving that portion of the matrix unchanged.
        /// This can save processing time when filling large areas with values that would
        /// not significantly alter the visual output.
        /// </summary>
        /// <typeparam name="T">The type of data stored in the matrix (e.g., byte, float).</typeparam>
        /// <param name="x">The x-coordinate of the starting position within the matrix.</param>
        /// <param name="y">The y-coordinate of the starting position within the matrix.</param>
        /// <param name="length">The number of elements to fill.</param>
        /// <param name="value">The value to fill the span with.</param>
        /// <param name="valueMinThreshold">The minimum threshold value for filling. If the value is below this threshold, the fill operation is skipped.</param>
        public void FillSpan<T>(int x, int y, int length, T value, T valueMinThreshold) where T : struct, IComparisonOperators<T, T, bool>
        {
            if (length <= 0 || value < valueMinThreshold) return; // Ignore threshold (mostly if blacks), spare cycles
            src.GetSpan<T>(length, src.GetPixelPos(x, y)).Fill(value);
        }

        /// <summary>
        /// Fills a span of the matrix data with a specified value, starting from a given position
        /// and extending for a specified length. The method includes an optimization to skip
        /// filling when the provided value is below a certain threshold, which is particularly
        /// useful for performance when dealing with low-intensity colors (e.g., near black).
        /// If the value is below the threshold, the method simply advances the start position by the
        /// length without modifying the data, effectively leaving that portion of the matrix unchanged.
        /// This can save processing time when filling large areas with values that would
        /// not significantly alter the visual output.
        /// </summary>
        /// <typeparam name="T">The type of data stored in the matrix (e.g., byte, float).</typeparam>
        /// <param name="position">The starting position within the matrix span.</param>
        /// <param name="length">The number of elements to fill.</param>
        /// <param name="value">The value to fill the span with.</param>
        /// <param name="valueMinThreshold">The minimum threshold value for filling. If the value is below this threshold, the fill operation is skipped.</param>

        public void FillSpan<T>(Point position, int length, T value, T valueMinThreshold) where T : struct, IComparisonOperators<T, T, bool>
        {
            src.FillSpan(position.X, position.Y, length, value, valueMinThreshold);
        }
        #endregion

        #region ROI methods
        /// <summary>
        /// Extracts a region of interest from the current matrix using the specified rectangle.
        /// </summary>
        /// <param name="roi">The rectangle that defines the region of interest to extract. The coordinates and size must be within the
        /// bounds of the current matrix.</param>
        /// <returns>A new matrix representing the specified region of interest.</returns>
        public Mat Roi(Rectangle roi)
        {
            return new Mat(src, roi);
        }

        /// <summary>
        /// Extracts a region of interest starting at the origin (0, 0) with the specified size.
        /// </summary>
        /// <param name="size">The size of the region of interest.</param>
        /// <returns>A new matrix representing the specified region of interest.</returns>
        public Mat Roi(Size size)
        {
            return new Mat(src, new(Point.Empty, size));
        }

        /// <summary>
        /// Extracts a region of interest starting at the origin (0, 0) with the size of the specified matrix.
        /// </summary>
        /// <param name="fromMat">The matrix whose size defines the region of interest.</param>
        /// <returns>A new matrix representing the specified region of interest.</returns>
        public Mat Roi(Mat fromMat)
        {
            ArgumentNullException.ThrowIfNull(fromMat);
            return new Mat(src, new(Point.Empty, fromMat.Size));
        }

        /// <summary>
        /// Calculates the bounding rectangle of non-zero pixels and returns the corresponding region of interest.
        /// </summary>
        /// <param name="boundingRectangle">The bounding rectangle of non-zero pixels.</param>
        /// <returns>A new matrix representing the bounding rectangle region.</returns>
        public Mat RoiFromBoundingRectangle(out Rectangle boundingRectangle)
        {
            if (src.IsEmpty)
            {
                boundingRectangle = Rectangle.Empty;
                return new Mat();
            }
            boundingRectangle = CvInvoke.BoundingRectangle(src);
            return new Mat(src, boundingRectangle);
        }

        /// <summary>
        /// Calculates the bounding rectangle of non-zero pixels and returns the corresponding region of interest.
        /// </summary>
        /// <returns>A new MatRoi representing the bounding rectangle region.</returns>
        public MatRoi RoiFromBoundingRectangle()
        {
            return MatRoi.CreateFromBoundingRectangle(src);
        }

        /// <summary>
        /// Extracts a region of interest centered within the source matrix with the specified size.
        /// </summary>
        /// <param name="size">The size of the centered region of interest.</param>
        /// <returns>A new matrix representing the centered region of interest.</returns>
        public Mat RoiFromCenter(Size size)
        {
            if (src.Size == size) return src.Roi(size);

            return src.SafeRoi(new Rectangle(
                src.Width / 2 - size.Width / 2,
                src.Height / 2 - size.Height / 2,
                size.Width,
                size.Height
            ));
        }

        /// <summary>
        /// Returns a region of interest (ROI) from the source matrix, expanding the specified rectangle by the given
        /// padding values while ensuring the resulting ROI remains within the matrix bounds.
        /// </summary>
        /// <remarks>If the requested padding would cause the ROI to exceed the matrix boundaries, the ROI
        /// is automatically clipped to fit within the source matrix. This method does not modify the original
        /// matrix.</remarks>
        /// <param name="roi">The rectangle that defines the initial region of interest within the source matrix. The rectangle is
        /// adjusted by the specified padding values.</param>
        /// <param name="outRoi">Returns the sanitized roi rectangle</param>
        /// <param name="padLeft">The number of pixels to add to the left side of the ROI. Must be zero or positive.</param>
        /// <param name="padTop">The number of pixels to add to the top side of the ROI. Must be zero or positive.</param>
        /// <param name="padRight">The number of pixels to add to the right side of the ROI. Must be zero or positive.</param>
        /// <param name="padBottom">The number of pixels to add to the bottom side of the ROI. Must be zero or positive.</param>
        /// <returns>A Mat object representing the adjusted region of interest, guaranteed to be within the bounds of the source
        /// matrix.</returns>
        public Mat SafeRoi(Rectangle roi, out Rectangle outRoi, int padLeft = 0, int padTop = 0, int padRight = 0,
            int padBottom = 0)
        {
            int x = Math.Max(0, roi.X - padLeft);
            int y = Math.Max(0, roi.Y - padTop);
            int right = Math.Min(src.Width, roi.Right + padRight);
            int bottom = Math.Min(src.Height, roi.Bottom + padBottom);
            int width = right - x;
            int height = bottom - y;
            if (width <= 0 || height <= 0)
            {
                outRoi = Rectangle.Empty;
                return new Mat();
            }
            outRoi = new Rectangle(x, y, width, height);
            return new Mat(src, outRoi);
        }

        /// <summary>
        /// Returns a region of interest (ROI) from the source matrix, expanding the specified rectangle by the given
        /// padding values while ensuring the resulting ROI remains within the matrix bounds.
        /// </summary>
        /// <remarks>If the requested padding would cause the ROI to exceed the matrix boundaries, the ROI
        /// is automatically clipped to fit within the source matrix. This method does not modify the original
        /// matrix.</remarks>
        /// <param name="roi">The rectangle that defines the initial region of interest within the source matrix. The rectangle is
        /// adjusted by the specified padding values.</param>
        /// <param name="padLeft">The number of pixels to add to the left side of the ROI. Must be zero or positive.</param>
        /// <param name="padTop">The number of pixels to add to the top side of the ROI. Must be zero or positive.</param>
        /// <param name="padRight">The number of pixels to add to the right side of the ROI. Must be zero or positive.</param>
        /// <param name="padBottom">The number of pixels to add to the bottom side of the ROI. Must be zero or positive.</param>
        /// <returns>A Mat object representing the adjusted region of interest, guaranteed to be within the bounds of the source
        /// matrix.</returns>
        public Mat SafeRoi(Rectangle roi, int padLeft = 0, int padTop = 0, int padRight = 0, int padBottom = 0)
        {
            return src.SafeRoi(roi, out _, padLeft, padTop, padRight, padBottom);
        }

        /// <summary>
        /// Calculates a safe region of interest (ROI) within the current matrix, applying the specified padding to
        /// ensure the ROI remains within the matrix boundaries.
        /// </summary>
        /// <remarks>If the requested ROI plus padding would exceed the matrix boundaries, the ROI is
        /// automatically adjusted to fit within the valid area. This method does not modify the original
        /// matrix.</remarks>
        /// <param name="roi">The rectangle that defines the initial region of interest to be adjusted.</param>
        /// <param name="outRoi">When this method returns, contains the adjusted rectangle representing the safe ROI after applying the
        /// specified padding.</param>
        /// <param name="padding">The amount of padding to apply to each edge of the region of interest, specified as a Size structure. The
        /// Width and Height represent the horizontal and vertical padding, respectively.</param>
        /// <returns>A Mat object representing the region of interest adjusted to fit safely within the matrix after applying the
        /// specified padding.</returns>
        public Mat SafeRoi(Rectangle roi, out Rectangle outRoi, Size padding)
        {
            return src.SafeRoi(roi, out outRoi, padding.Width, padding.Height, padding.Width, padding.Height);
        }

        /// <summary>
        /// Returns a new Mat that represents a region of interest (ROI) within the current Mat, expanded by the
        /// specified padding.
        /// </summary>
        /// <remarks>If the requested ROI and padding extend beyond the boundaries of the original Mat,
        /// the resulting region will be clipped to fit within the source Mat.</remarks>
        /// <param name="roi">A Rectangle that defines the region of interest to extract from the current Mat.</param>
        /// <param name="padding">A Size specifying the amount of horizontal and vertical padding to add to the ROI. The width and height of
        /// the padding are applied to the respective sides of the ROI.</param>
        /// <returns>A Mat containing the region of interest, including the specified padding. The returned Mat will not exceed
        /// the bounds of the original Mat.</returns>
        public Mat SafeRoi(Rectangle roi, Size padding)
        {
            return src.SafeRoi(roi, out _, padding.Width, padding.Height, padding.Width, padding.Height);
        }

        /// <summary>
        /// Calculates a safe region of interest (ROI) within the current matrix, applying the specified padding to
        /// ensure the ROI remains within the matrix boundaries.
        /// </summary>
        /// <remarks>If the requested ROI plus padding would exceed the matrix boundaries, the ROI is
        /// automatically adjusted to fit within the valid area. This method does not modify the original
        /// matrix.</remarks>
        /// <param name="roi">The rectangle that defines the initial region of interest to be adjusted.</param>
        /// <param name="outRoi">When this method returns, contains the adjusted rectangle representing the safe ROI after applying the
        /// specified padding.</param>
        /// <param name="padding">The amount of padding to apply to each edge of the region of interest, specified as an integer. The
        /// same value is applied to all sides of the ROI.</param>
        /// <returns>A Mat object representing the region of interest adjusted to fit safely within the matrix after applying the
        /// specified padding.</returns>
        public Mat SafeRoi(Rectangle roi, out Rectangle outRoi, int padding)
        {
            return src.SafeRoi(roi, out outRoi, padding, padding, padding, padding);
        }

        /// <summary>
        /// Returns a new Mat that represents a region of interest (ROI) within the current Mat, expanded by the
        /// specified padding.
        /// </summary>
        /// <remarks>If the requested ROI and padding extend beyond the boundaries of the original Mat,
        /// the resulting region will be clipped to fit within the source Mat.</remarks>
        /// <param name="roi">A Rectangle that defines the region of interest to extract from the current Mat.</param>
        /// <param name="padding">An integer specifying the amount of padding to add to each side of the ROI.</param>
        /// <returns>A Mat containing the region of interest, including the specified padding. The returned Mat will not exceed
        /// the bounds of the original Mat.</returns>
        public Mat SafeRoi(Rectangle roi, int padding)
        {
            return src.SafeRoi(roi, out _, padding, padding, padding, padding);
        }
        #endregion

        #region Pixel accessors

        /// <summary>
        /// Calculates the byte count for a given number of pixels, based on the matrix's element size (channels x depth).
        /// </summary>
        /// <param name="pixels">The number of pixels to calculate the byte count for.</param>
        /// <returns>The total byte count for the specified number of pixels.</returns>
        public int GetByteCount(int pixels)
        {
            return pixels * src.ElementSize;
        }

        /// <summary>
        /// Calculates the byte count for a given number of pixels, based on the matrix's element size (channels x depth).
        /// </summary>
        /// <param name="pixels">The number of pixels to calculate the byte count for.</param>
        /// <returns>The total byte count for the specified number of pixels.</returns>
        public long GetByteCount(long pixels)
        {
            return pixels * src.ElementSize;
        }

        /// <summary>
        /// Gets the byte offset of the specified row within the matrix data.
        /// </summary>
        /// <param name="y">The row index.</param>
        /// <returns>The byte offset to the start of the specified row.</returns>
        public int GetRowPos(int y)
        {
            return y * src.RealStep;
        }

        /// <summary>
        /// Gets a pixel index position on a span given X and Y
        /// </summary>
        /// <param name="x">X coordinate</param>
        /// <param name="y">Y coordinate</param>
        /// <returns>The pixel index position</returns>
        public int GetPixelPos(int x, int y)
        {
            return src.GetRowPos(y) + src.GetByteCount(x);
        }

        /// <summary>
        /// Gets a pixel index position on a span given X and Y
        /// </summary>
        /// <param name="point">X and Y Location</param>
        /// <returns>The pixel index position</returns>
        public int GetPixelPos(Point point)
        {
            return src.GetPixelPos(point.X, point.Y);
        }

        /// <summary>
        /// Gets a byte pixel at a position
        /// </summary>
        /// <param name="pos"></param>
        /// <returns></returns>
        public byte GetByte(int pos)
        {
            if (src.IsEmpty || src.DataPointer == IntPtr.Zero)
                throw new InvalidOperationException("Cannot read from an empty or uninitialized Mat.");
            unsafe
            {
                return *(src.BytePointer + pos);
            }
        }

        /// <summary>
        /// Gets a byte pixel at a position
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        public byte GetByte(int x, int y) => src.GetByte(src.GetPixelPos(x, y));

        /// <summary>
        /// Gets a byte pixel at a position
        /// </summary>
        /// <param name="pos"></param>
        /// <returns></returns>
        public byte GetByte(Point pos) => src.GetByte(src.GetPixelPos(pos.X, pos.Y));

        /// <summary>
        /// Sets a byte pixel at a position
        /// </summary>
        /// <param name="pixel"></param>
        /// <param name="value"></param>
        public void SetByte(int pixel, byte value)
        {
            if (src.IsEmpty || src.DataPointer == IntPtr.Zero)
                throw new InvalidOperationException("Cannot write to an empty or uninitialized Mat.");
            unsafe
            {
                *(src.BytePointer + pixel) = value;
            }
        }

        /// <summary>
        /// Sets a byte pixel at a position
        /// </summary>
        /// <param name="pixel"></param>
        /// <param name="value"></param>
        public void SetByte(int pixel, byte[] value)
        {
            ArgumentNullException.ThrowIfNull(value);
            if (src.IsEmpty || src.DataPointer == IntPtr.Zero)
                throw new InvalidOperationException("Cannot write to an empty or uninitialized Mat.");
            Marshal.Copy(value, 0, src.DataPointer + pixel, value.Length);
        }

        /// <summary>
        /// Sets a byte pixel at a position
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="value"></param>
        public void SetByte(int x, int y, byte value) => src.SetByte(src.GetPixelPos(x, y), value);

        /// <summary>
        /// Sets a byte pixel at a position
        /// </summary>
        /// <param name="pos"></param>
        /// <param name="value"></param>
        public void SetByte(Point pos, byte value) => src.SetByte(src.GetPixelPos(pos.X, pos.Y), value);

        /// <summary>
        /// Sets a byte pixel at a position
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="value"></param>
        public void SetByte(int x, int y, byte[] value) => src.SetByte(src.GetPixelPos(x, y), value);

        /// <summary>
        /// Returns a copy of contents of the underlying data as a byte array.
        /// </summary>
        /// <returns>A byte array containing the raw data.</returns>
        public byte[] ToArray()
        {
            var length = src.LengthInt32;
            if (length == 0) return [];
            var copy = GC.AllocateUninitializedArray<byte>(length);
            src.CopyTo(copy);
            return copy;
        }
        #endregion

        #region Find methods

        /// <summary>
        /// Finds the first negative (Black) pixel
        /// </summary>
        /// <param name="startPos">Start pixel position</param>
        /// <param name="length">Pixel span length</param>
        /// <returns>Pixel position in the span, or -1 if not found</returns>
        public int FindFirstNegativePixel<T>(int startPos = 0, int length = 0) where T : struct, INumber<T>, IMinMaxValue<T>
        {
            return src.FindFirstPixelEqualTo(T.Zero, startPos, length);
        }

        /// <summary>
        /// Finds the first positive pixel
        /// </summary>
        /// <param name="startPos">Start pixel position</param>
        /// <param name="length">Pixel span length</param>
        /// <returns>Pixel position in the span, or -1 if not found</returns>
        public int FindFirstPositivePixel<T>(int startPos = 0, int length = 0) where T : struct, INumber<T>, IMinMaxValue<T>
        {
            return src.FindFirstPixelGreaterThan(T.Zero, startPos, length);
        }

        /// <summary>
        /// Finds the first pixel that is <paramref name="value"/>
        /// </summary>
        /// <param name="value"></param>
        /// <param name="startPos">Start pixel position</param>
        /// <param name="length">Pixel span length</param>
        /// <returns>Pixel position in the span, or -1 if not found</returns>
        public int FindFirstPixelEqualTo<T>(T value, int startPos = 0, int length = 0) where T : struct
        {
            var span = src.GetReadOnlySpan<T>(length, startPos);
            var found = span.IndexOf(value, null);
            return found < 0 ? -1 : startPos + found;
        }

        /// <summary>
        /// Finds the first pixel that is at less than <paramref name="value"/>
        /// </summary>
        /// <param name="value"></param>
        /// <param name="startPos">Start pixel position</param>
        /// <param name="length">Pixel span length</param>
        /// <returns>Pixel position in the span, or -1 if not found</returns>
        public int FindFirstPixelLessThan<T>(T value, int startPos = 0, int length = 0) where T : struct, INumber<T>, IMinMaxValue<T>
        {
            if (value == T.MinValue) return -1;
            var span = src.GetReadOnlySpan<T>(length, startPos);
            var found = span.IndexOfAnyInRange(T.MinValue, value - T.One);
            return found < 0 ? -1 : startPos + found;
        }

        /// <summary>
        /// Finds the first pixel that is at less or equal than <paramref name="value"/>
        /// </summary>
        /// <param name="value"></param>
        /// <param name="startPos">Start pixel position</param>
        /// <param name="length">Pixel span length</param>
        /// <returns>Pixel position in the span, or -1 if not found</returns>
        public int FindFirstPixelEqualOrLessThan<T>(T value, int startPos = 0, int length = 0) where T : struct, INumber<T>, IMinMaxValue<T>
        {
            var span = src.GetReadOnlySpan<T>(length, startPos);
            var found = span.IndexOfAnyInRange(T.MinValue, value);
            return found < 0 ? -1 : startPos + found;
        }

        /// <summary>
        /// Finds the first pixel that is at greater than <paramref name="value"/>
        /// </summary>
        /// <param name="value"></param>
        /// <param name="startPos">Start pixel position</param>
        /// <param name="length">Pixel span length</param>
        /// <returns>Pixel position in the span, or -1 if not found</returns>
        public int FindFirstPixelGreaterThan<T>(T value, int startPos = 0, int length = 0) where T : struct, INumber<T>, IMinMaxValue<T>
        {
            if (value == T.MaxValue) return -1;
            var span = src.GetReadOnlySpan<T>(length, startPos);
            var found = span.IndexOfAnyInRange(value + T.One, T.MaxValue);
            return found < 0 ? -1 : startPos + found;
        }

        /// <summary>
        /// Finds the first pixel that is at equal or greater than <paramref name="value"/>
        /// </summary>
        /// <param name="value"></param>
        /// <param name="startPos">Start pixel position</param>
        /// <param name="length">Pixel span length</param>
        /// <returns>Pixel position in the span, or -1 if not found</returns>
        public int FindFirstPixelEqualOrGreaterThan<T>(T value, int startPos = 0, int length = 0) where T : struct, INumber<T>, IMinMaxValue<T>
        {
            var span = src.GetReadOnlySpan<T>(length, startPos);
            var found = span.IndexOfAnyInRange(value, T.MaxValue);
            return found < 0 ? -1 : startPos + found;
        }

        /// <summary>
        /// Scan sequential strides of continuous pixels
        /// </summary>
        /// <param name="strideLimit">Size limit of a single stride.</param>
        /// <param name="breakOnRows">True to break the stride sequence on a new row, otherwise false.</param>
        /// <param name="startOnFirstPositivePixel">True to skip the first sequence of black pixels, otherwise false.</param>
        /// <param name="excludeBlacks">True to exclude black strides from returning, otherwise false.</param>
        /// <param name="thresholdGrey">Value to threshold the grey, below or equal to this value will set to 0, otherwise <paramref name="thresholdMaxGrey"/></param>
        /// <param name="thresholdMaxGrey">Grey value to set when the threshold is above the limit.</param>
        /// <returns></returns>
        public List<GreyStride> ScanStrides(int strideLimit = 0, bool breakOnRows = false, bool startOnFirstPositivePixel = false, bool excludeBlacks = false, byte thresholdGrey = 0, byte thresholdMaxGrey = byte.MaxValue)
        {
            ArgumentOutOfRangeException.ThrowIfNotEqual(src.NumberOfChannels, 1);
            ArgumentOutOfRangeException.ThrowIfNegative(strideLimit);
            ArgumentOutOfRangeException.ThrowIfEqual(strideLimit, 1);

            var result = new List<GreyStride>();

            if (src.IsEmpty) return result;

            int i = 0;
            int x = 0;
            int y = 0;

            int index = 0;
            Point location = default;
            uint stride = 0;
            byte grey = 0;

            var maxWidth = src.Width;
            var useThreshold = thresholdGrey is > byte.MinValue and < byte.MaxValue;

            var span = src.GetReadOnlySpan<byte>();

            if (excludeBlacks || startOnFirstPositivePixel)
            {
                for (; i < span.Length; i++)
                {
                    grey = span[i];
                    if (useThreshold)
                    {
                        grey = grey <= thresholdGrey ? byte.MinValue : thresholdMaxGrey;
                    }
                    if (grey == 0) continue;
                    index = i;
                    location.X = i % maxWidth;
                    location.Y = y = i / maxWidth;
                    stride = 1;
                    i++;

                    x = location.X + 1;

                    break;
                }
            }

            for (; i < span.Length; i++)
            {
                // Check for rows
                if (x == maxWidth)
                {
                    y++;

                    if (breakOnRows && stride > 0)
                    {
                        if (!excludeBlacks || grey > 0) result.Add(new GreyStride(index, location, stride, grey));
                        index = i;
                        location.X = 0;
                        location.Y = y;
                        stride = 1;
                        grey = span[i];
                        if (useThreshold)
                        {
                            grey = grey <= thresholdGrey ? byte.MinValue : thresholdMaxGrey;
                        }

                        x = 1;

                        continue;
                    }

                    x = 0;
                }

                // Check for sequence
                var currentGrey = span[i];
                if (useThreshold)
                {
                    currentGrey = currentGrey <= thresholdGrey ? byte.MinValue : thresholdMaxGrey;
                }

                if (currentGrey == grey)
                {
                    stride++;
                    if (stride == strideLimit)
                    {
                        if (!excludeBlacks || grey > 0) result.Add(new GreyStride(index, location, stride, grey));
                        index = i + 1;
                        location.X = index % maxWidth;
                        location.Y = index / maxWidth;
                        stride = 0;
                    }
                }
                else
                {
                    if (stride > 0)
                    {
                        if (!excludeBlacks || grey > 0) result.Add(new GreyStride(index, location, stride, grey));
                        index = i;
                        location.X = x;
                        location.Y = y;
                    }

                    stride = 1;
                    grey = currentGrey;
                }

                x++;
            }

            // Return the left over
            if (stride > 0 && (!excludeBlacks || grey > 0))
            {
                result.Add(new GreyStride(index, location, stride, grey));
            }

            return result;
        }

        /// <summary>
        /// Scan sequential strides of continuous pixels
        /// </summary>
        /// <param name="greyFunc">Function to filter and process the gray value.</param>
        /// <param name="strideLimit">Size limit of a single stride.</param>
        /// <param name="breakOnRows">True to break the stride sequence on a new row, otherwise false.</param>
        /// <param name="startOnFirstPositivePixel">True to skip the first sequence of black pixels, otherwise false.</param>
        /// <param name="excludeBlacks">True to exclude black strides from returning, otherwise false.</param>
        /// <returns></returns>
        public List<GreyStride> ScanStrides(Func<byte, byte> greyFunc, int strideLimit = 0, bool breakOnRows = false, bool startOnFirstPositivePixel = false, bool excludeBlacks = false)
        {
            ArgumentNullException.ThrowIfNull(greyFunc);
            ArgumentOutOfRangeException.ThrowIfNotEqual(src.NumberOfChannels, 1);
            ArgumentOutOfRangeException.ThrowIfNegative(strideLimit);
            ArgumentOutOfRangeException.ThrowIfEqual(strideLimit, 1);

            var result = new List<GreyStride>();

            if (src.IsEmpty) return result;

            int i = 0;
            int x = 0;
            int y = 0;

            int index = 0;
            Point location = default;
            uint stride = 0;
            byte grey = 0;

            var maxWidth = src.Width;

            var span = src.GetReadOnlySpan<byte>();

            if (excludeBlacks || startOnFirstPositivePixel)
            {
                for (; i < span.Length; i++)
                {
                    grey = greyFunc(span[i]);

                    if (grey == 0) continue;
                    index = i;
                    location.X = i % maxWidth;
                    location.Y = y = i / maxWidth;
                    stride = 1;
                    i++;

                    x = location.X + 1;

                    break;
                }
            }

            for (; i < span.Length; i++)
            {
                // Check for rows
                if (x == maxWidth)
                {
                    y++;

                    if (breakOnRows && stride > 0)
                    {
                        if (!excludeBlacks || grey > 0) result.Add(new GreyStride(index, location, stride, grey));
                        index = i;
                        location.X = 0;
                        location.Y = y;
                        stride = 1;
                        grey = greyFunc(span[i]);

                        x = 1;

                        continue;
                    }

                    x = 0;
                }

                // Check for sequence
                var currentGrey = greyFunc(span[i]);

                if (currentGrey == grey)
                {
                    stride++;
                    if (stride == strideLimit)
                    {
                        if (!excludeBlacks || grey > 0) result.Add(new GreyStride(index, location, stride, grey));
                        index = i + 1;
                        location.X = index % maxWidth;
                        location.Y = index / maxWidth;
                        stride = 0;
                    }
                }
                else
                {
                    if (stride > 0)
                    {
                        if (!excludeBlacks || grey > 0) result.Add(new GreyStride(index, location, stride, grey));
                        index = i;
                        location.X = x;
                        location.Y = y;
                    }

                    stride = 1;
                    grey = currentGrey;
                }

                x++;
            }

            // Return the left over
            if (stride > 0 && (!excludeBlacks || grey > 0))
            {
                result.Add(new GreyStride(index, location, stride, grey));
            }

            return result;
        }

        /// <summary>
        /// Scan sequential lines in X or Y direction
        /// </summary>
        /// <param name="vertically">True to scan vertically, otherwise horizontally</param>
        /// <param name="thresholdGrey">Value to threshold the grey, less or equal to this value will set to 0, otherwise 255</param>
        /// <param name="offset">Value to offset the coordinates with.</param>
        /// <returns>List of all lines</returns>
        public List<GreyLine> ScanLines(bool vertically = false, byte thresholdGrey = 0, Point offset = default)
        {
            ArgumentOutOfRangeException.ThrowIfNotEqual(src.NumberOfChannels, 1);

            var lines = new List<GreyLine>();

            if (src.IsEmpty) return lines;

            var matSize = src.Size;
            var useThreshold = thresholdGrey is > byte.MinValue and < byte.MaxValue;

            GreyLine line = default;

            byte grey;
            int x;
            int y;

            if (vertically)
            {
                var span = src.GetReadOnlySpan2D<byte>();
                for (x = 0; x < matSize.Width; x++)
                {
                    line.StartX = x + offset.X;
                    line.StartY = offset.Y;
                    line.EndX = x + offset.X;
                    line.EndY = offset.Y;
                    line.Grey = 0;

                    for (y = 0; y < matSize.Height; y++)
                    {
                        grey = span[y, x];
                        if (useThreshold)
                        {
                            grey = grey <= thresholdGrey ? byte.MinValue : byte.MaxValue;
                        }

                        if (line.Grey == 0)
                        {
                            if (grey == 0) continue;
                            line.StartY = y + offset.Y;
                            line.Grey = grey;
                            continue;
                        }

                        if (grey == line.Grey) continue;
                        line.EndY = y - 1 + offset.Y;
                        lines.Add(line);

                        line.Grey = 0;
                        y--;
                    }

                    if (line.Grey > 0)
                    {
                        line.EndY = y - 1 + offset.Y;
                        lines.Add(line);
                    }
                }
            }
            else // Horizontal
            {
                for (y = 0; y < matSize.Height; y++)
                {
                    var span = src.GetRowSpan<byte>(y);
                    line.StartX = offset.X;
                    line.StartY = y + offset.Y;
                    line.EndX = offset.X;
                    line.EndY = y + offset.Y;
                    line.Grey = 0;

                    for (x = 0; x < matSize.Width; x++)
                    {
                        grey = span[x];
                        if (useThreshold)
                        {
                            grey = grey <= thresholdGrey ? byte.MinValue : byte.MaxValue;
                        }

                        if (line.Grey == 0)
                        {
                            if (grey == 0) continue;
                            line.StartX = x + offset.X;
                            line.Grey = grey;
                            continue;
                        }

                        if (grey == line.Grey) continue;
                        line.EndX = x - 1 + offset.X;
                        lines.Add(line);

                        line.Grey = 0;
                        x--;
                    }

                    if (line.Grey > 0)
                    {
                        line.EndX = x - 1 + offset.X;
                        lines.Add(line);
                    }
                }
            }

            return lines;
        }

        /// <summary>
        /// Scan sequential lines in X or Y direction
        /// </summary>
        /// <param name="greyFunc">Function to filter and process the gray value</param>
        /// <param name="vertically">True to scan vertically, otherwise horizontally</param>
        /// <param name="offset">Value to offset the coordinates with.</param>
        /// <returns>List of all lines</returns>
        public List<GreyLine> ScanLines(Func<byte, byte> greyFunc, bool vertically = false, Point offset = default)
        {
            ArgumentNullException.ThrowIfNull(greyFunc);
            ArgumentOutOfRangeException.ThrowIfNotEqual(src.NumberOfChannels, 1);

            var lines = new List<GreyLine>();

            if (src.IsEmpty) return lines;
            var matSize = src.Size;

            GreyLine line = default;

            byte grey;
            int x;
            int y;

            if (vertically)
            {
                var span = src.GetReadOnlySpan2D<byte>();
                for (x = 0; x < matSize.Width; x++)
                {
                    line.StartX = x + offset.X;
                    line.StartY = offset.Y;
                    line.EndX = x + offset.X;
                    line.EndY = offset.Y;
                    line.Grey = 0;

                    for (y = 0; y < matSize.Height; y++)
                    {
                        grey = greyFunc(span[y, x]);

                        if (line.Grey == 0)
                        {
                            if (grey == 0) continue;
                            line.StartY = y + offset.Y;
                            line.Grey = grey;
                            continue;
                        }

                        if (grey == line.Grey) continue;
                        line.EndY = y - 1 + offset.Y;
                        lines.Add(line);

                        line.Grey = 0;
                        y--;
                    }

                    if (line.Grey > 0)
                    {
                        line.EndY = y - 1 + offset.Y;
                        lines.Add(line);
                    }
                }
            }
            else
            {
                for (y = 0; y < matSize.Height; y++)
                {
                    var span = src.GetRowSpan<byte>(y);
                    line.StartX = offset.X;
                    line.StartY = y + offset.Y;
                    line.EndX = offset.X;
                    line.EndY = y + offset.Y;
                    line.Grey = 0;

                    for (x = 0; x < matSize.Width; x++)
                    {
                        grey = greyFunc(span[x]);

                        if (line.Grey == 0)
                        {
                            if (grey == 0) continue;
                            line.StartX = x + offset.X;
                            line.Grey = grey;
                            continue;
                        }

                        if (grey == line.Grey) continue;
                        line.EndX = x - 1 + offset.X;
                        lines.Add(line);

                        line.Grey = 0;
                        x--;
                    }

                    if (line.Grey > 0)
                    {
                        line.EndX = x - 1 + offset.X;
                        lines.Add(line);
                    }
                }
            }

            return lines;
        }
        #endregion

        #region Create methods

        /// <summary>
        /// Creates a binary mask from the specified contours by drawing them filled with white on a black background.
        /// </summary>
        /// <param name="contours">The contours to draw on the mask.</param>
        /// <param name="offset">An optional offset applied to all contour points.</param>
        /// <returns>A new single-channel matrix with the contour regions filled in white.</returns>
        public Mat CreateMask(VectorOfVectorOfPoint contours, Point offset = default)
        {
            var mask = src.NewZeros();
            CvInvoke.DrawContours(mask, contours, -1, WhiteColor, -1, LineType.EightConnected, null, int.MaxValue, offset);
            return mask;
        }

        /// <summary>
        /// Creates a binary mask from the specified contours by drawing them filled with white on a black background.
        /// </summary>
        /// <param name="contours">The contours to draw on the mask.</param>
        /// <param name="offset">An optional offset applied to all contour points.</param>
        /// <returns>A new single-channel matrix with the contour regions filled in white.</returns>
        public Mat CreateMask(Point[][] contours, Point offset = default)
        {
            using var vec = new VectorOfVectorOfPoint(contours);
            return src.CreateMask(vec, offset);
        }

        /// <summary>
        /// Crops the matrix to the bounding rectangle of its non-zero pixels.
        /// </summary>
        /// <param name="cloneInsteadRoi">If <see langword="true"/>, returns a cloned matrix; otherwise returns a ROI that shares memory with the source.</param>
        /// <returns>A new matrix cropped to the bounding rectangle of non-zero content.</returns>
        public Mat CropByBounds(bool cloneInsteadRoi = false)
        {
            var rect = CvInvoke.BoundingRectangle(src);
            if (rect.Size == Size.Empty) return src.New();
            if (src.Size == rect.Size) return cloneInsteadRoi ? src.Clone() : src.Roi(src.Size);
            var roi = src.Roi(rect);

            if (cloneInsteadRoi)
            {
                var clone = roi.Clone();
                roi.Dispose();
                return clone;
            }

            return roi;
        }

        /// <summary>
        /// Crops the matrix to its bounding rectangle with a uniform margin added around the content.
        /// </summary>
        /// <param name="margin">The margin in pixels to add on all sides.</param>
        /// <returns>A new matrix cropped and padded with the specified margin.</returns>
        public Mat CropByBounds(int margin)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(margin);

            return src.CropByBounds(new Size(margin, margin));
        }

        /// <summary>
        /// Crops the matrix to its bounding rectangle with the specified horizontal and vertical margins.
        /// </summary>
        /// <param name="margin">The margin size where Width is horizontal padding and Height is vertical padding.</param>
        /// <returns>A new matrix cropped and padded with the specified margins.</returns>
        public Mat CropByBounds(Size margin)
        {
            if (margin.Width < 0 || margin.Height < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(margin), margin,
                    "Margin width and height must be non-negative.");
            }

            var rect = CvInvoke.BoundingRectangle(src);
            if (rect.Size == Size.Empty) return src.New();
            using var roi = src.Size == rect.Size ? src.Roi(src.Size) : src.Roi(rect);

            var numberOfChannels = roi.NumberOfChannels;
            var cropped = InitMat(new Size(roi.Width + margin.Width * 2, roi.Height + margin.Height * 2), numberOfChannels, roi.Depth);

            using var dest = new Mat(cropped, new Rectangle(margin.Width, margin.Height, roi.Width, roi.Height));
            roi.CopyTo(dest);

            return cropped;
        }

        /// <summary>
        /// Crops the matrix to its bounding rectangle and copies the result into the destination matrix.
        /// </summary>
        /// <param name="dst">The destination matrix, which is resized to fit the cropped result.</param>
        public void CropByBounds(Mat dst)
        {
            using var mat = src.CropByBounds();
            dst.Create(mat.Rows, mat.Cols, mat.Depth, mat.NumberOfChannels);
            mat.CopyTo(dst);
        }

        /// <summary>
        /// Creates a new <see cref="Mat"/> with the specified padding added around each edge.
        /// </summary>
        /// <param name="top">Padding in pixels to add to the top edge.</param>
        /// <param name="bottom">Padding in pixels to add to the bottom edge.</param>
        /// <param name="left">Padding in pixels to add to the left edge.</param>
        /// <param name="right">Padding in pixels to add to the right edge.</param>
        /// <param name="borderType">The border extrapolation method.</param>
        /// <param name="value">The border value when <paramref name="borderType"/> is <see cref="BorderType.Constant"/>.</param>
        /// <returns>A new <see cref="Mat"/> with the added padding.</returns>
        public Mat Pad(int top, int bottom, int left, int right, BorderType borderType = BorderType.Constant, MCvScalar value = default)
        {
            var dst = new Mat();
            CvInvoke.CopyMakeBorder(src, dst, top, bottom, left, right, borderType, value);
            return dst;
        }

        /// <summary>
        /// Creates a new <see cref="Mat"/> with uniform padding added around all edges.
        /// </summary>
        /// <param name="padding">Padding in pixels to add to each edge.</param>
        /// <param name="borderType">The border extrapolation method.</param>
        /// <param name="value">The border value when <paramref name="borderType"/> is <see cref="BorderType.Constant"/>.</param>
        /// <returns>A new <see cref="Mat"/> with the added padding.</returns>
        public Mat Pad(int padding, BorderType borderType = BorderType.Constant, MCvScalar value = default)
            => src.Pad(padding, padding, padding, padding, borderType, value);

        #endregion

        #region Image format methods

        /// <summary>
        /// Generates SVG path strings from the contours found in the matrix. Tags are not included.
        /// </summary>
        /// <param name="compression">The contour approximation method.</param>
        /// <param name="threshold">If <see langword="true"/>, applies a binary threshold before extracting contours.</param>
        /// <returns>An IEnumerable of SVG path data strings.</returns>
        public IEnumerable<string> GetSvgPath(ChainApproxMethod compression = ChainApproxMethod.ChainApproxSimple, bool threshold = true)
        {
            Mat mat = src;
            try
            {
                if (threshold)
                {
                    mat = new Mat();
                    CvInvoke.Threshold(src, mat, 127, byte.MaxValue, ThresholdType.Binary);
                }

                using var contours = mat.FindContours(out var hierarchy, RetrType.Tree, compression);

                var sb = new StringBuilder(256);
                for (int i = 0; i < contours.Size; i++)
                {
                    // Cache the inner contour to avoid repeated native interop indexer calls
                    var contour = contours[i];

                    if (hierarchy[i, EmguContour.HierarchyParent] == -1) // Top hierarchy
                    {
                        if (sb.Length > 0)
                        {
                            yield return sb.ToString();
                            sb.Clear();
                        }
                    }
                    else
                    {
                        sb.Append(' ');
                    }

                    var firstPoint = contour[0];
                    sb.Append('M')
                        .Append(' ')
                        .Append(firstPoint.X)
                        .Append(' ')
                        .Append(firstPoint.Y)
                        .Append(" L");

                    var contourSize = contour.Size;
                    for (int x = 1; x < contourSize; x++)
                    {
                        var pt = contour[x];
                        sb.Append(' ')
                            .Append(pt.X)
                            .Append(' ')
                            .Append(pt.Y);
                    }
                    sb.Append(" Z");
                }

                if (sb.Length > 0)
                {
                    yield return sb.ToString();
                }
            }
            finally
            {
                if (!ReferenceEquals(src, mat)) mat.Dispose();
            }
        }
        #endregion

        #region Letterbox methods

        /// <summary>
        /// Creates a new image with the specified target width and height by resizing the current image to fit within
        /// the target dimensions while maintaining aspect ratio, and adding padding (letterboxing) as needed.
        /// </summary>
        /// <remarks>The aspect ratio of the original image is preserved. Padding is added equally on both
        /// sides to center the image within the target dimensions. The returned Mat must be disposed by the caller when
        /// no longer needed.</remarks>
        /// <param name="targetWidth">The desired width, in pixels, of the output image. Must be greater than zero.</param>
        /// <param name="targetHeight">The desired height, in pixels, of the output image. Must be greater than zero.</param>
        /// <param name="paddingColor">The color to use for the padding areas added to the image. Typically used to fill the regions not covered by
        /// the resized image.</param>
        /// <param name="scale">When this method returns, contains the scale factor applied to the original image to fit it within the
        /// target dimensions.</param>
        /// <param name="padX">When this method returns, contains the number of pixels of horizontal padding added to the left side of the
        /// image.</param>
        /// <param name="padY">When this method returns, contains the number of pixels of vertical padding added to the top of the image.</param>
        /// <returns>A new Mat object representing the letterboxed image with the specified target size and padding color.</returns>
        public Mat CreateLetterBox(int targetWidth, int targetHeight, MCvScalar paddingColor,
            out float scale, out int padX, out int padY)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(targetWidth);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(targetHeight);

            var originalSize = src.Size;
            var targetSize = new Size(targetWidth, targetHeight);

            scale = 1f;
            padX = 0;
            padY = 0;

            if (originalSize == targetSize)
                return src.Clone();

            // Calculate letterbox parameters
            // Find scale that fits the image within targetSize while maintaining aspect ratio
            scale = Math.Min((float)targetWidth / originalSize.Width, (float)targetHeight / originalSize.Height);
            int newWidth = Math.Max(1, (int)(originalSize.Width * scale));
            int newHeight = Math.Max(1, (int)(originalSize.Height * scale));

            // Calculate padding to center the image
            padX = (targetWidth - newWidth) / 2;
            padY = (targetHeight - newHeight) / 2;

            // Create letterboxed image
            using Mat resized = new();
            CvInvoke.Resize(src, resized, new Size(newWidth, newHeight), 0, 0, Inter.Linear);

            // Create padded image
            var letterboxed = new Mat(targetSize, src.Depth, src.NumberOfChannels);
            letterboxed.SetTo(paddingColor); // color padding

            // Copy resized image to center of letterboxed image
            var roi = new Rectangle(padX, padY, newWidth, newHeight);
            using var roiMat = letterboxed.SafeRoi(roi);
            resized.CopyTo(roiMat);

            return letterboxed;
        }

        /// <summary>
        /// Resizes the current image to fit within the specified target dimensions while preserving the aspect ratio,
        /// adding black padding as needed to fill the remaining space (letterboxing).
        /// </summary>
        /// <remarks>The original image is scaled to fit within the target dimensions without distortion.
        /// Black padding is added to the top, bottom, left, or right as needed to maintain the aspect ratio. The
        /// returned image will always have the exact target width and height.</remarks>
        /// <param name="targetWidth">The width, in pixels, of the target output image. Must be a positive integer.</param>
        /// <param name="targetHeight">The height, in pixels, of the target output image. Must be a positive integer.</param>
        /// <param name="scale">When this method returns, contains the scaling factor applied to the original image to fit within the target
        /// dimensions.</param>
        /// <param name="padX">When this method returns, contains the number of pixels of horizontal padding added to center the image
        /// within the target width.</param>
        /// <param name="padY">When this method returns, contains the number of pixels of vertical padding added to center the image within
        /// the target height.</param>
        /// <returns>A new Mat object containing the letterboxed image with the specified target dimensions.</returns>
        public Mat CreateLetterBox(int targetWidth, int targetHeight, out float scale, out int padX, out int padY)
        {
            return src.CreateLetterBox(targetWidth, targetHeight, new MCvScalar(0, 0, 0, 255), out scale, out padX,
                out padY);
        }

        #endregion

        #region Transform methods
        /// <summary>
        /// Applies an affine transformation to the current image, scaling it by the specified factors and translating it
        /// </summary>
        /// <param name="xScale"></param>
        /// <param name="yScale"></param>
        /// <param name="xTrans"></param>
        /// <param name="yTrans"></param>
        /// <param name="dstSize"></param>
        /// <param name="interpolation"></param>
        public void Transform(double xScale, double yScale, double xTrans = 0, double yTrans = 0, Size dstSize = default, Inter interpolation = Inter.Linear)
        {
            //var dst = new Mat(src.Size, src.Depth, src.NumberOfChannels);
            using var translateTransform = new Matrix<double>(2, 3);
            translateTransform[0, 0] = xScale; // xScale
            translateTransform[1, 1] = yScale; // yScale
            translateTransform[0, 2] = xTrans; //x translation + compensation of x scaling
            translateTransform[1, 2] = yTrans; // y translation + compensation of y scaling
            CvInvoke.WarpAffine(src, src, translateTransform, dstSize.IsEmpty ? src.Size : dstSize, interpolation);
        }

        /// <summary>
        /// Rotates a Mat by an angle while keeping the image size
        /// </summary>
        /// <param name="angle">Angle in degrees to rotate</param>
        /// <param name="newSize"></param>
        /// <param name="scale"></param>
        public void RotateFromCenter(double angle, Size newSize = default, double scale = 1.0) => src.RotateFromCenter(src, angle, newSize, scale);

        /// <summary>
        /// Rotates a Mat by an angle while keeping the image size
        /// </summary>
        /// <param name="dst"></param>
        /// <param name="angle"></param>
        /// <param name="newSize"></param>
        /// <param name="scale"></param>
        public void RotateFromCenter(Mat dst, double angle, Size newSize = default, double scale = 1.0)
        {
            if (angle % 360 == 0 && Math.Abs(scale - 1.0) < 0.001)
            {
                if (!ReferenceEquals(src, dst))
                {
                    src.CopyTo(dst);
                }

                return;
            }
            if (newSize.IsEmpty)
            {
                newSize = src.Size;
            }

            var halfWidth = src.Width / 2.0f;
            var halfHeight = src.Height / 2.0f;
            using var translateTransform = new Matrix<double>(2, 3);
            CvInvoke.GetRotationMatrix2D(new PointF(halfWidth, halfHeight), -angle, scale, translateTransform);

            if (src.Size != newSize)
            {
                // adjust the rotation matrix to take into account translation
                translateTransform[0, 2] += newSize.Width / 2.0 - halfWidth;
                translateTransform[1, 2] += newSize.Height / 2.0 - halfHeight;
            }

            CvInvoke.WarpAffine(src, dst, translateTransform, newSize);
        }

        /// <summary>
        /// Rotates a Mat by an angle while adjusting bounds to fit the rotated content
        /// </summary>
        /// <param name="angle"></param>
        /// <param name="scale"></param>
        public void RotateAdjustBounds(double angle, double scale = 1.0) => src.RotateAdjustBounds(src, angle, scale);

        /// <summary>
        /// Rotates a Mat by an angle while adjusting bounds to fit the rotated content
        /// </summary>
        /// <param name="dst"></param>
        /// <param name="angle"></param>
        /// <param name="scale"></param>
        public void RotateAdjustBounds(Mat dst, double angle, double scale = 1.0)
        {
            if (angle % 360 == 0 && Math.Abs(scale - 1.0) < 0.001)
            {
                if (!ReferenceEquals(src, dst))
                {
                    src.CopyTo(dst);
                }

                return;
            }

            var halfWidth = src.Width / 2.0f;
            var halfHeight = src.Height / 2.0f;
            using var translateTransform = new Matrix<double>(2, 3);
            CvInvoke.GetRotationMatrix2D(new PointF(halfWidth, halfHeight), -angle, scale, translateTransform);
            var cos = Math.Abs(translateTransform[0, 0]);
            var sin = Math.Abs(translateTransform[0, 1]);

            // compute the new bounding dimensions of the image
            int newWidth = (int)(src.Height * sin + src.Width * cos);
            int newHeight = (int)(src.Height * cos + src.Width * sin);

            // adjust the rotation matrix to take into account translation
            translateTransform[0, 2] += newWidth / 2.0 - halfWidth;
            translateTransform[1, 2] += newHeight / 2.0 - halfHeight;


            CvInvoke.WarpAffine(src, dst, translateTransform, new Size(newWidth, newHeight));
        }

        /// <summary>
        /// Scale image from it center, preserving src bounds
        /// https://stackoverflow.com/a/62543674/933976
        /// </summary>
        /// <param name="xScale">X scale factor</param>
        /// <param name="yScale">Y scale factor</param>
        /// <param name="xTrans">X translation</param>
        /// <param name="yTrans">Y translation</param>
        /// <param name="dstSize">Destination size</param>
        /// <param name="interpolation">Interpolation mode</param>
        public void TransformFromCenter(double xScale, double yScale, double xTrans = 0, double yTrans = 0, Size dstSize = default, Inter interpolation = Inter.Linear)
        {
            src.Transform(xScale, yScale,
                xTrans + (src.Width - src.Width * xScale) / 2.0,
                yTrans + (src.Height - src.Height * yScale) / 2.0, dstSize, interpolation);
        }

        /// <summary>
        /// Resize source src proportional to a scale
        /// </summary>
        /// <param name="scale"></param>
        /// <param name="interpolation"></param>
        public void Resize(double scale, Inter interpolation = Inter.Linear)
        {
            if (Math.Abs(scale - 1) < 0.001) return;
            CvInvoke.Resize(src, src, new Size((int)(src.Width * scale), (int)(src.Height * scale)), 0, 0, interpolation);
        }

        /// <summary>
        /// Shrinks the image to fit within the specified maximum dimensions while preserving
        /// the aspect ratio. Unlike <c>CreateLetterBox</c>, no padding is added and smaller images are not upscaled.
        /// </summary>
        /// <param name="maxWidth">The maximum width, in pixels, of the output image.</param>
        /// <param name="maxHeight">The maximum height, in pixels, of the output image.</param>
        /// <param name="interpolation">The interpolation method to use when resizing.</param>
        /// <returns><see langword="true"/> if the image was shrunk; otherwise <see langword="false"/>.</returns>
        public bool ShrinkToFitPreserveAspect(int maxWidth, int maxHeight, Inter interpolation = Inter.Linear)
        {
            var scale = Math.Min((double)maxWidth / src.Width, (double)maxHeight / src.Height);
            if (scale >= 1.0) return false;

            CvInvoke.Resize(src, src,
                new Size((int)(src.Width * scale), (int)(src.Height * scale)),
                0, 0, interpolation);
            return true;
        }

        /// <summary>
        /// Shrinks the image to fit within the specified maximum dimensions while preserving
        /// the aspect ratio. Unlike <c>CreateLetterBox</c>, no padding is added and smaller images are not upscaled.
        /// </summary>
        /// <param name="dst">The destination matrix to receive the shrunken result. If the source already fits, the source is copied to <paramref name="dst"/>.</param>
        /// <param name="maxWidth">The maximum width, in pixels, of the output image.</param>
        /// <param name="maxHeight">The maximum height, in pixels, of the output image.</param>
        /// <param name="interpolation">The interpolation method to use when resizing.</param>
        /// <returns><see langword="true"/> if the image was shrunk; otherwise <see langword="false"/>.</returns>
        public bool ShrinkToFitPreserveAspect(Mat dst, int maxWidth, int maxHeight, Inter interpolation = Inter.Linear)
        {
            var scale = Math.Min((double)maxWidth / src.Width, (double)maxHeight / src.Height);
            if (scale >= 1.0)
            {
                if (!ReferenceEquals(src, dst))
                {
                    src.CopyTo(dst);
                }

                return false;
            }

            CvInvoke.Resize(src, dst,
                new Size((int)(src.Width * scale), (int)(src.Height * scale)),
                0, 0, interpolation);
            return true;
        }

        /// <summary>
        /// Flips the matrix around the specified axis in-place.
        /// </summary>
        /// <param name="flipType">The axis around which to flip.</param>
        public void Flip(FlipType flipType) => CvInvoke.Flip(src, src, flipType);

        /// <summary>
        /// Flips the matrix around the specified axis into a destination matrix.
        /// </summary>
        /// <param name="dst">The destination matrix to receive the flipped result.</param>
        /// <param name="flipType">The axis around which to flip.</param>
        public void Flip(Mat dst, FlipType flipType) => CvInvoke.Flip(src, dst, flipType);
        #endregion

        #region Draw methods

        /// <summary>
        /// Draws a line between two points with thickness compensation for more accurate rendering.
        /// </summary>
        /// <param name="pt1">The first point of the line.</param>
        /// <param name="pt2">The second point of the line.</param>
        /// <param name="color">The color of the line.</param>
        /// <param name="thickness">The thickness of the line in pixels.</param>
        /// <param name="lineType">The type of line to draw.</param>
        public void DrawLineAccurate(Point pt1, Point pt2, MCvScalar color, int thickness, LineType lineType = LineType.EightConnected)
        {
            if (thickness >= 3)
            {
                thickness--;
            }

            CvInvoke.Line(src, pt1, pt2, color, thickness, lineType);
        }

        /// <summary>
        /// Draws a rotated square around a center point.
        /// </summary>
        /// <param name="size">The side length of the square in pixels.</param>
        /// <param name="center">The center point of the square.</param>
        /// <param name="color">The color of the square.</param>
        /// <param name="angle">The rotation angle in degrees.</param>
        /// <param name="thickness">The thickness of the outline. Use -1 for a filled square.</param>
        /// <param name="lineType">The type of line to draw.</param>
        public void DrawRotatedSquare(int size, Point center, MCvScalar color, int angle = 0, int thickness = -1, LineType lineType = LineType.EightConnected)
            => src.DrawRotatedRectangle(new(size, size), center, color, angle, thickness, lineType);

        /// <summary>
        /// Draws a rotated rectangle around a center point.
        /// </summary>
        /// <param name="size">The size of the rectangle.</param>
        /// <param name="center">The center point of the rectangle.</param>
        /// <param name="color">The color of the rectangle.</param>
        /// <param name="angle">The rotation angle in degrees.</param>
        /// <param name="thickness">The thickness of the outline. Use -1 for a filled rectangle.</param>
        /// <param name="lineType">The type of line to draw.</param>
        public void DrawRotatedRectangle(Size size, Point center, MCvScalar color, int angle = 0, int thickness = -1, LineType lineType = LineType.EightConnected)
        {
            if (angle == 0)
            {
                src.DrawCenteredRectangle(size, center, color, thickness, lineType);
                return;
            }

            var rect = new RotatedRect(center, size, angle);
            var vertices = rect.GetVertices();
            var points = new Point[vertices.Length];

            for (int i = 0; i < vertices.Length; i++)
            {
                points[i] = new(
                    (int)Math.Round(vertices[i].X),
                    (int)Math.Round(vertices[i].Y)
                );
            }

            if (thickness <= 0)
            {
                using var vec = new VectorOfPoint(points);
                CvInvoke.FillConvexPoly(src, vec, color, lineType);
            }
            else
            {
                CvInvoke.Polylines(src, points, true, color, thickness, lineType);
            }
        }

        /// <summary>
        /// Draws a square centered around a point.
        /// </summary>
        /// <param name="size">The side length of the square in pixels.</param>
        /// <param name="center">The center point of the square.</param>
        /// <param name="color">The color of the square.</param>
        /// <param name="thickness">The thickness of the outline. Use -1 for a filled square.</param>
        /// <param name="lineType">The type of line to draw.</param>
        public void DrawCenteredSquare(int size, Point center, MCvScalar color, int thickness = -1, LineType lineType = LineType.EightConnected)
            => src.DrawCenteredRectangle(new Size(size, size), center, color, thickness, lineType);

        /// <summary>
        /// Draws a rectangle centered around a point.
        /// </summary>
        /// <param name="size">The size of the rectangle.</param>
        /// <param name="center">The center point of the rectangle.</param>
        /// <param name="color">The color of the rectangle.</param>
        /// <param name="thickness">The thickness of the outline. Use -1 for a filled rectangle.</param>
        /// <param name="lineType">The type of line to draw.</param>
        public void DrawCenteredRectangle(Size size, Point center, MCvScalar color, int thickness = -1, LineType lineType = LineType.EightConnected)
        {
            center.Offset(size.Width / -2, size.Height / -2);
            CvInvoke.Rectangle(src, new Rectangle(center, size), color, thickness, lineType);
        }

        /// <summary>
        /// Draws a regular polygon with the specified number of sides and diameter.
        /// </summary>
        /// <param name="sides">Number of polygon sides. Special: use 1 to draw a line and >= 100 to draw a native OpenCV circle.</param>
        /// <param name="diameter">The diameter for both X and Y axes.</param>
        /// <param name="center">The center position of the polygon.</param>
        /// <param name="color">The color of the polygon.</param>
        /// <param name="startingAngle">The starting rotation angle in degrees.</param>
        /// <param name="thickness">The thickness of the outline. Use -1 for a filled polygon.</param>
        /// <param name="lineType">The type of line to draw.</param>
        /// <param name="flip">An optional flip transformation to apply.</param>
        /// <param name="midpointRounding">The rounding mode for vertex coordinates.</param>
        public void DrawPolygon(int sides, SizeF diameter, PointF center, MCvScalar color, double startingAngle = 0, int thickness = -1, LineType lineType = LineType.EightConnected, FlipType? flip = null, MidpointRounding midpointRounding = MidpointRounding.AwayFromZero)
        {
            if (sides == 1)
            {
                var point1 = center with { X = MathF.Round(center.X - diameter.Width / 2, midpointRounding) };
                var point2 = point1 with { X = point1.X + diameter.Width - 1 };
                point1 = point1.Rotate(startingAngle, center);
                point2 = point2.Rotate(startingAngle, center);

                if (flip is FlipType.Horizontal or FlipType.Both)
                {
                    var newPoint1 = new PointF(point2.X, point1.Y);
                    var newPoint2 = new PointF(point1.X, point2.Y);
                    point1 = newPoint1;
                    point2 = newPoint2;
                }

                if (flip is FlipType.Vertical or FlipType.Both)
                {
                    var newPoint1 = new PointF(point1.X, point2.Y);
                    var newPoint2 = new PointF(point2.X, point1.Y);
                    point1 = newPoint1;
                    point2 = newPoint2;
                }

                CvInvoke.Line(src, point1.ToPoint(midpointRounding), point2.ToPoint(midpointRounding), color, thickness < 1 ? 1 : thickness, lineType);
                return;
            }
            if (sides >= 100)
            {
                src.DrawCircle(center.ToPoint(midpointRounding),
                    new Size((int)Math.Round(diameter.Width / 2, midpointRounding), (int)Math.Round(diameter.Height / 2, midpointRounding)),
                    color, -1, lineType);
                return;
            }

            var points = DrawingExtensions.GetPolygonVertices(sides, diameter, center, startingAngle,
                flip is FlipType.Horizontal or FlipType.Both,
                flip is FlipType.Vertical or FlipType.Both,
                midpointRounding);

            if (thickness <= 0)
            {
                using var vec = new VectorOfPoint(points);
                CvInvoke.FillConvexPoly(src, vec, color, lineType);
            }
            else
            {
                CvInvoke.Polylines(src, points, true, color, thickness, lineType);
            }
        }

        /// <summary>
        /// Draw a polygon given number of sides and diameter
        /// </summary>
        /// <param name="sides">Number of polygon sides, Special: use 1 to draw a line and >= 100 to draw a native OpenCV circle</param>
        /// <param name="diameter">Diameter</param>
        /// <param name="center">Center position</param>
        /// <param name="color"></param>
        /// <param name="startingAngle"></param>
        /// <param name="thickness"></param>
        /// <param name="lineType"></param>
        /// <param name="flip"></param>
        /// <param name="midpointRounding"></param>
        public void DrawPolygon(int sides, float diameter, PointF center, MCvScalar color, double startingAngle = 0, int thickness = -1, LineType lineType = LineType.EightConnected, FlipType? flip = null, MidpointRounding midpointRounding = MidpointRounding.AwayFromZero)
        {
            src.DrawPolygon(sides, new SizeF(diameter, diameter), center, color, startingAngle, thickness, lineType, flip, midpointRounding);
        }

        /// <summary>
        /// Draw a polygon given number of sides and diameter (Aligned in X axis)
        /// </summary>
        /// <param name="sides">Number of polygon sides, Special: use 1 to draw a line and >= 100 to draw a native OpenCV circle</param>
        /// <param name="diameter">Diameter for both X and Y axis</param>
        /// <param name="center">Center position</param>
        /// <param name="color"></param>
        /// <param name="startingAngle"></param>
        /// <param name="thickness"></param>
        /// <param name="lineType"></param>
        /// <param name="flip"></param>
        /// <param name="midpointRounding"></param>
        public void DrawAlignedPolygon(int sides, SizeF diameter, PointF center, MCvScalar color, double startingAngle = 0, int thickness = -1, LineType lineType = LineType.EightConnected, FlipType? flip = null, MidpointRounding midpointRounding = MidpointRounding.AwayFromZero)
        {
            if (sides >= 3 && sides != 4)
            {
                startingAngle += (180 - (360.0 / sides)) / 2;
            }

            src.DrawPolygon(sides, diameter, center, color, startingAngle, thickness, lineType, flip, midpointRounding);
        }

        /// <summary>
        /// Draws a regular polygon with the specified number of sides, diameter, and center, aligned according to the
        /// given starting angle and optional flip transformation.
        /// </summary>
        /// <remarks>If the number of sides is 4, the polygon is not drawn. The polygon is aligned so that
        /// its first vertex is placed according to the specified starting angle, and the remaining vertices are
        /// distributed evenly around the center.</remarks>
        /// <param name="sides">The number of sides for the polygon. Must be 3 or greater, but not equal to 4.</param>
        /// <param name="diameter">The diameter of the polygon, measured from one vertex to the opposite vertex.</param>
        /// <param name="center">The center point of the polygon in image coordinates.</param>
        /// <param name="color">The color used to draw the polygon.</param>
        /// <param name="startingAngle">The angle, in degrees, at which the first vertex of the polygon is placed, measured counterclockwise from
        /// the horizontal axis. Defaults to 0.</param>
        /// <param name="thickness">The thickness of the polygon's outline. A value of -1 fills the polygon.</param>
        /// <param name="lineType">The type of the line used to draw the polygon's edges.</param>
        /// <param name="flip">An optional flip transformation to apply to the polygon before drawing. If null, no flip is applied.</param>
        /// <param name="midpointRounding">Specifies how to round midpoint values when calculating vertex positions.</param>
        public void DrawAlignedPolygon(int sides, float diameter, PointF center, MCvScalar color, double startingAngle = 0, int thickness = -1, LineType lineType = LineType.EightConnected, FlipType? flip = null, MidpointRounding midpointRounding = MidpointRounding.AwayFromZero)
        {
            if (sides >= 3 && sides != 4)
            {
                startingAngle += (180 - (360.0 / sides)) / 2;
            }

            src.DrawPolygon(sides, new SizeF(diameter, diameter), center, color, startingAngle, thickness, lineType, flip, midpointRounding);
        }

        /// <summary>
        /// Draw a circle with a center point and radius
        /// </summary>
        /// <param name="center"></param>
        /// <param name="radius"></param>
        /// <param name="color"></param>
        /// <param name="thickness"></param>
        /// <param name="lineType"></param>
        public void DrawCircle(Point center, Size radius, MCvScalar color, int thickness = -1, LineType lineType = LineType.EightConnected)
        {
            if (Math.Abs(radius.Width - radius.Height) < 0.01)
            {
                CvInvoke.Circle(src, center, radius.Width, color, thickness, lineType);
            }
            else
            {
                CvInvoke.Ellipse(src, center, radius, 0, 0, 360, color, thickness, lineType);
            }
        }

        /// <summary>
        /// Draw a circle with a center point and radius
        /// </summary>
        /// <param name="center"></param>
        /// <param name="radius"></param>
        /// <param name="color"></param>
        /// <param name="thickness"></param>
        /// <param name="lineType"></param>
        public void DrawCircle(Point center, int radius, MCvScalar color, int thickness = -1, LineType lineType = LineType.EightConnected)
        {
            CvInvoke.Circle(src, center, radius, color, thickness, lineType);
        }
        #endregion

        #region Text methods
        /// <summary>
        /// Extended OpenCV PutText to accepting line breaks, line alignment and rotation
        /// </summary>
        public void PutTextRotated(string text, Point org, FontFace fontFace, double fontScale,
            MCvScalar color,
            int thickness = 1, LineType lineType = LineType.EightConnected, bool bottomLeftOrigin = false,
            PutTextLineAlignment lineAlignment = default, double angle = 0)
            => src.PutTextRotated(text, org, fontFace, fontScale, color, thickness, 0, lineType, bottomLeftOrigin,
                lineAlignment, angle);

        /// <summary>
        /// Extended OpenCV PutText to accepting line breaks, line alignment and rotation
        /// </summary>
        public void PutTextRotated(string text, Point org, FontFace fontFace, double fontScale, MCvScalar color,
            int thickness = 1, int lineGapOffset = 0, LineType lineType = LineType.EightConnected, bool bottomLeftOrigin = false, PutTextLineAlignment lineAlignment = default, double angle = 0)
        {
            if (angle % 360 == 0) // No rotation needed, cheaper cycle
            {
                src.PutTextExtended(text, org, fontFace, fontScale, color, thickness, lineGapOffset, lineType, bottomLeftOrigin, lineAlignment);
                return;
            }

            using var rotatedSrc = src.Clone();
            rotatedSrc.RotateAdjustBounds(-angle);
            org.Offset((rotatedSrc.Width - src.Width) / 2, (rotatedSrc.Height - src.Height) / 2);
            org = org.Rotate(-angle, new Point(rotatedSrc.Size.Width / 2, rotatedSrc.Size.Height / 2));
            rotatedSrc.PutTextExtended(text, org, fontFace, fontScale, color, thickness, lineGapOffset, lineType, bottomLeftOrigin, lineAlignment);

            using var mask = rotatedSrc.NewZeros();
            mask.PutTextExtended(text, org, fontFace, fontScale, WhiteColor, thickness, lineGapOffset, lineType, bottomLeftOrigin, lineAlignment);

            rotatedSrc.RotateFromCenter(angle, src.Size);
            mask.RotateFromCenter(angle, src.Size);

            rotatedSrc.CopyTo(src, mask);
        }
        #endregion

        #region Utility methods

        /// <summary>
        /// Calculates the 64-bit xxHash3 hash value of the underlying data source.
        /// </summary>
        /// <remarks>The xxHash3 algorithm provides fast, non-cryptographic hashing suitable for checksums
        /// and hash-based data structures. The result is deterministic for the same input data.</remarks>
        /// <returns>A 64-bit unsigned integer representing the xxHash3 hash of the data.</returns>
        public ulong GetXxHash3()
        {
            return XxHash3.HashToUInt64(src.GetSpan<byte>());
        }

        /// <summary>
        /// Performs morphological skeletonization on the source image, reducing it to a single-pixel-wide representation
        /// of its shape while preserving the topological structure.
        /// </summary>
        /// <param name="iterations">When this method returns, contains the number of iterations performed to complete the skeletonization.</param>
        /// <param name="ksize">The size of the structuring element. Defaults to 3x3 if empty.</param>
        /// <param name="elementShape">The shape of the morphological structuring element.</param>
        /// <param name="cancellationToken">A token to cancel the operation.</param>
        /// <returns>A new matrix containing the skeletonized image.</returns>
        public Mat Skeletonize(out int iterations, Size ksize = default, MorphShapes elementShape = MorphShapes.Rectangle, CancellationToken cancellationToken = default)
        {
            if (ksize.IsEmpty) ksize = new Size(3, 3);
            var skeleton = src.NewZeros();
            using var kernel = CvInvoke.GetStructuringElement(elementShape, ksize, AnchorCenter);
            using var current = src.Clone();
            using var eroded = new Mat();
            using var temp = new Mat();

            iterations = 0;
            while (true)
            {
                cancellationToken.ThrowIfCancellationRequested();
                iterations++;

                // erode and dilate the image using the structuring element
                CvInvoke.Erode(current, eroded, kernel, AnchorCenter, 1, BorderType.Reflect101, default);
                CvInvoke.Dilate(eroded, temp, kernel, AnchorCenter, 1, BorderType.Reflect101, default);

                // subtract the temporary image from the original, eroded
                // image, then take the bitwise 'or' between the skeleton
                // and the temporary image
                CvInvoke.Subtract(current, temp, temp);
                CvInvoke.BitwiseOr(skeleton, temp, skeleton);

                // if there are no more 'white' pixels in the image, then
                // break from the loop
                if (!CvInvoke.HasNonZero(eroded)) break;

                // reuse the existing buffer instead of cloning
                eroded.CopyTo(current);
            }

            return skeleton;
        }

        /// <summary>
        /// Performs morphological skeletonization on the source image, reducing it to a single-pixel-wide representation
        /// </summary>
        /// <param name="ksize">The size of the structuring element. Defaults to 3x3 if empty.</param>
        /// <param name="elementShape">The shape of the morphological structuring element.</param>
        /// <param name="cancellationToken">A token to cancel the operation.</param>
        /// <returns>A new matrix containing the skeletonized image.</returns>
        public Mat Skeletonize(Size ksize = default, MorphShapes elementShape = MorphShapes.Rectangle, CancellationToken cancellationToken = default)
            => src.Skeletonize(out _, ksize, elementShape, cancellationToken);

        /// <summary>
        /// Gets the minimum and maximum pixel values and their locations in the matrix.
        /// </summary>
        /// <returns>A tuple containing the minimum value, maximum value, and their respective locations.</returns>
        public (double Min, double Max, Point MinLocation, Point MaxLocation) GetMinMax()
        {
            double min = 0, max = 0;
            Point minLoc = default, maxLoc = default;
            CvInvoke.MinMaxLoc(src, ref min, ref max, ref minLoc, ref maxLoc);
            return (min, max, minLoc, maxLoc);
        }

        /// <summary>
        /// Splits the matrix into its individual channels.
        /// </summary>
        /// <returns>An array of single-channel <see cref="Mat"/> objects. The caller must dispose each element.</returns>
        public Mat[] SplitChannels()
        {
            using var channels = new VectorOfMat();
            CvInvoke.Split(src, channels);
            var result = new Mat[channels.Size];
            for (var i = 0; i < channels.Size; i++)
                result[i] = channels[i];
            return result;
        }

        /// <summary>
        /// Extracts a single channel from the matrix by index.
        /// </summary>
        /// <param name="index">The zero-based channel index to extract.</param>
        /// <returns>A new single-channel <see cref="Mat"/> containing the extracted channel data.</returns>
        public Mat GetChannel(int index)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(index);
            if (index >= src.NumberOfChannels)
                throw new ArgumentOutOfRangeException(nameof(index), index, $"Index must be less than the number of channels ({src.NumberOfChannels}).");
            var dst = new Mat();
            CvInvoke.ExtractChannel(src, dst, index);
            return dst;
        }
        #endregion
    }
}
