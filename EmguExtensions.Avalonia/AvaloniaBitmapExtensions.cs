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

using Avalonia.Media.Imaging;
using Avalonia.Platform;
using CommunityToolkit.HighPerformance;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Point = System.Drawing.Point;
using PixelPoint = Avalonia.PixelPoint;
using PixelRect = Avalonia.PixelRect;
using Rectangle = System.Drawing.Rectangle;

namespace EmguExtensions.Avalonia;

/// <summary>
/// Extension methods for Avalonia Bitmap types.
/// </summary>
/// <remarks>
/// Everything that exposes pixel memory (spans, <see cref="Mat"/> views, addresses) is only valid while the
/// <see cref="ILockedFramebuffer"/> it comes from is locked (not disposed), do not keep it past that point.
/// </remarks>
public static class AvaloniaBitmapExtensions
{
    #region Validation helpers

    private static void ValidatePixelCoordinates(ILockedFramebuffer framebuffer, int x, int y)
    {
        if ((uint)x >= (uint)framebuffer.Size.Width)
            throw new ArgumentOutOfRangeException(nameof(x), x, $"X must be in the range [0, {framebuffer.Size.Width}).");
        if ((uint)y >= (uint)framebuffer.Size.Height)
            throw new ArgumentOutOfRangeException(nameof(y), y, $"Y must be in the range [0, {framebuffer.Size.Height}).");
    }

    private static void ValidateRoi(ILockedFramebuffer framebuffer, Rectangle roi)
    {
        if (roi.X < 0 || roi.Y < 0 || roi.Width < 0 || roi.Height < 0 || roi.Right > framebuffer.Size.Width ||
            roi.Bottom > framebuffer.Size.Height)
            throw new ArgumentOutOfRangeException(nameof(roi), roi,
                $"The region of interest (ROI) must be within the bounds of the bitmap (width: {framebuffer.Size.Width}, height: {framebuffer.Size.Height}).");
    }

    private static Rectangle ToRectangle(PixelRect rect) => new(rect.X, rect.Y, rect.Width, rect.Height);

    private static BitmapInfo CreateBitmapInfo(ILockedFramebuffer framebuffer, bool includeAddress)
    {
        return new BitmapInfo
        {
            Address = includeAddress ? framebuffer.Address : IntPtr.Zero,
            Width = framebuffer.Size.Width,
            Height = framebuffer.Size.Height,
            RowBytes = framebuffer.RowBytes,
            BytesPerPixel = framebuffer.BytesPerPixel,
            IsContiguous = framebuffer.IsContinuous,
        };
    }

    #endregion

    extension(ILockedFramebuffer src)
    {
        #region Properties

        /// <summary>
        /// Gets the number of bytes per pixel for the framebuffer's <see cref="ILockedFramebuffer.Format"/>.
        /// </summary>
        public int BytesPerPixel => src.Format.BitsPerPixel / 8;

        /// <summary>
        /// Gets the total number of bytes in the source, including any per-row padding.
        /// </summary>
        /// <exception cref="OverflowException">The size does not fit in a 32-bit integer, use <c>ByteCountInt64</c>.</exception>
        public int ByteCount => checked(src.RowBytes * src.Size.Height);

        /// <summary>
        /// Gets the total number of bytes in the source, including any per-row padding, as a 64-bit integer.
        /// </summary>
        public long ByteCountInt64 => (long)src.RowBytes * src.Size.Height;

        /// <summary>
        /// Gets the total number of pixels in the locked framebuffer, calculated as width multiplied by height.
        /// </summary>
        /// <exception cref="OverflowException">The count does not fit in a 32-bit integer.</exception>
        public int PixelCount => checked(src.Size.Width * src.Size.Height);

        /// <summary>
        /// Gets a value indicating whether the framebuffer rows are stored contiguously, i.e. the row stride
        /// (<see cref="ILockedFramebuffer.RowBytes"/>) contains no trailing padding.
        /// </summary>
        public bool IsContinuous => src.RowBytes == src.Size.Width * src.BytesPerPixel;

        #endregion

        #region Mat views

        /// <summary>
        /// Creates a <see cref="Mat"/> that is a zero-copy view over the framebuffer memory, honoring the row stride.
        /// </summary>
        /// <remarks>
        /// The channels are in the order of the framebuffer <see cref="ILockedFramebuffer.Format"/> (for <c>Bgra8888</c> it is the OpenCV BGRA order,
        /// for <c>Rgba8888</c> it is RGBA). Writes to the Mat change the framebuffer.
        /// The Mat does not own the memory: dispose it BEFORE disposing the framebuffer and never use it afterwards.
        /// </remarks>
        /// <returns>A 8-bit Mat with 1, 3 or 4 channels, or an empty Mat when the framebuffer has no pixels.</returns>
        /// <exception cref="NotSupportedException">The pixel format is not 8 bits per channel with 1, 3 or 4 channels.</exception>
        public Mat ToMat()
        {
            var channels = src.BytesPerPixel;
            if (channels is not (1 or 3 or 4))
                throw new NotSupportedException(
                    $"Only framebuffers with 1, 3 or 4 bytes per pixel can be viewed as a Mat, but the framebuffer uses {src.Format.BitsPerPixel}-bit pixels.");

            if (src.Size.Width <= 0 || src.Size.Height <= 0 || src.Address == IntPtr.Zero)
                return new Mat();

            return new Mat(src.Size.Height, src.Size.Width, DepthType.Cv8U, channels, src.Address, src.RowBytes);
        }

        /// <summary>
        /// Gets a <see cref="BitmapInfo"/> struct containing detailed information about the locked framebuffer, including its address.
        /// </summary>
        /// <returns>A <see cref="BitmapInfo"/> struct, its <see cref="BitmapInfo.Address"/> is only valid while the framebuffer is locked.</returns>
        public BitmapInfo GetBitmapInfo() => CreateBitmapInfo(src, true);

        #endregion

        #region Pixel positions

        /// <summary>
        /// Calculates the linear byte position for a pixel at the specified coordinates.
        /// </summary>
        /// <param name="x">The horizontal coordinate (column) of the pixel.</param>
        /// <param name="y">The vertical coordinate (row) of the pixel.</param>
        /// <returns>The byte offset position of the pixel in the buffer (taking the row stride into account).</returns>
        /// <exception cref="ArgumentOutOfRangeException">The coordinates are outside the bitmap.</exception>
        public int GetPixelBytePos(int x, int y)
        {
            ValidatePixelCoordinates(src, x, y);
            return checked(src.RowBytes * y + x * src.BytesPerPixel);
        }

        /// <summary>
        /// Calculates the linear byte position for a pixel at the specified coordinates.
        /// </summary>
        /// <param name="location">The point containing the X and Y coordinates.</param>
        /// <returns>The byte offset position of the pixel in the buffer (taking the row stride into account).</returns>
        /// <exception cref="ArgumentOutOfRangeException">The coordinates are outside the bitmap.</exception>
        public int GetPixelBytePos(Point location) => src.GetPixelBytePos(location.X, location.Y);

        /// <summary>
        /// Calculates the linear byte position for a pixel at the specified coordinates.
        /// </summary>
        /// <param name="location">The point containing the X and Y coordinates.</param>
        /// <returns>The byte offset position of the pixel in the buffer (taking the row stride into account).</returns>
        /// <exception cref="ArgumentOutOfRangeException">The coordinates are outside the bitmap.</exception>
        public int GetPixelBytePos(PixelPoint location) => src.GetPixelBytePos(location.X, location.Y);

        /// <summary>
        /// Calculates the pixel index of the specified coordinates in the flat pixel sequence (width as the row length).
        /// </summary>
        /// <remarks>It matches the indexes of <c>GetSpan</c>, which is only available for continuous framebuffers (see <c>IsContinuous</c>). Use <c>GetPixelBytePos</c> for padded framebuffers.</remarks>
        /// <param name="x">The horizontal coordinate (column) of the pixel.</param>
        /// <param name="y">The vertical coordinate (row) of the pixel.</param>
        /// <returns>The pixel offset position in the flat pixel sequence.</returns>
        /// <exception cref="ArgumentOutOfRangeException">The coordinates are outside the bitmap.</exception>
        public int GetPixelPos(int x, int y)
        {
            ValidatePixelCoordinates(src, x, y);
            return checked(src.Size.Width * y + x);
        }

        /// <summary>
        /// Calculates the pixel index of the specified coordinates in the flat pixel sequence (width as the row length).
        /// </summary>
        /// <param name="location">The point containing the X and Y coordinates.</param>
        /// <returns>The pixel position.</returns>
        /// <exception cref="ArgumentOutOfRangeException">The coordinates are outside the bitmap.</exception>
        public int GetPixelPos(Point location) => src.GetPixelPos(location.X, location.Y);

        /// <summary>
        /// Calculates the pixel index of the specified coordinates in the flat pixel sequence (width as the row length).
        /// </summary>
        /// <param name="location">The point containing the X and Y coordinates.</param>
        /// <returns>The pixel position.</returns>
        /// <exception cref="ArgumentOutOfRangeException">The coordinates are outside the bitmap.</exception>
        public int GetPixelPos(PixelPoint location) => src.GetPixelPos(location.X, location.Y);

        #endregion

        #region Flat span accessors

        /// <summary>
        /// Gets a span of bytes over the whole framebuffer memory (including any per-row padding).
        /// </summary>
        /// <param name="length">The number of bytes to include in the span. If 0, uses everything from the offset to the end of the framebuffer.</param>
        /// <param name="offset">The byte offset from the start of the source memory.</param>
        /// <returns>A span of bytes at the specified offset and length.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when the offset or length is negative or out of range.</exception>
        public Span<byte> GetSpanOfBytes(int length = 0, int offset = 0)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(length);
            ArgumentOutOfRangeException.ThrowIfNegative(offset);

            var maxLength = src.ByteCount - offset;

            if (maxLength < 0)
                throw new ArgumentOutOfRangeException(nameof(offset), offset, "Offset value overflow this Bitmap size.");

            if (length == 0)
            {
                length = maxLength;
            }
            else if (length > maxLength)
            {
                throw new ArgumentOutOfRangeException(nameof(length), length, $"The maximum size allowed for this Bitmap with an offset of {offset} is {maxLength}.");
            }

            unsafe
            {
                return new(IntPtr.Add(src.Address, offset).ToPointer(), length);
            }
        }

        /// <summary>
        /// Gets a read-only span of bytes over the whole framebuffer memory (including any per-row padding).
        /// </summary>
        /// <param name="length">The number of bytes to include in the span. If 0, uses everything from the offset to the end of the framebuffer.</param>
        /// <param name="offset">The byte offset from the start of the source memory.</param>
        /// <returns>A read-only span of bytes at the specified offset and length.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when the offset or length is negative or out of range.</exception>
        public ReadOnlySpan<byte> GetReadOnlySpanOfBytes(int length = 0, int offset = 0) => src.GetSpanOfBytes(length, offset);

        /// <summary>
        /// Gets a span of pixels from the source memory, where each pixel is represented as a 32-bit unsigned integer (e.g., BGRA format).
        /// </summary>
        /// <remarks>Only valid when the framebuffer is continuous (see <c>IsContinuous</c>). For padded buffers use <c>GetSpan2D</c> instead.</remarks>
        /// <param name="length">The number of pixels to include in the span. If 0, uses everything from the offset to the last pixel.</param>
        /// <param name="offset">The pixel offset from the start of the source memory.</param>
        /// <returns>A span of pixels at the specified offset and length.</returns>
        /// <exception cref="NotSupportedException">Thrown when the framebuffer is not continuous or its pixels are not 32-bit.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when the offset or length is negative or out of range.</exception>
        public Span<uint> GetSpan(int length = 0, int offset = 0)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(length);
            ArgumentOutOfRangeException.ThrowIfNegative(offset);

            if (src.BytesPerPixel != sizeof(uint))
                throw new NotSupportedException($"GetSpan<uint> requires a 32-bit pixel format, but the framebuffer uses {src.Format.BitsPerPixel}-bit pixels.");
            if (!src.IsContinuous)
                throw new NotSupportedException("To create a flat pixel Span, the framebuffer memory must be continuous (no row padding). Use Span2D instead.");

            var maxLength = src.PixelCount - offset;

            if (maxLength < 0)
                throw new ArgumentOutOfRangeException(nameof(offset), offset, "Offset value overflow this Bitmap size.");

            if (length == 0)
            {
                length = maxLength;
            }
            else if (length > maxLength)
            {
                throw new ArgumentOutOfRangeException(nameof(length), length, $"The maximum size allowed for this Bitmap with an offset of {offset} is {maxLength}.");
            }

            unsafe
            {
                return new(IntPtr.Add(src.Address, checked(offset * sizeof(uint))).ToPointer(), length);
            }
        }

        /// <summary>
        /// Gets a read-only span of pixels from the source memory, where each pixel is represented as a 32-bit unsigned integer (e.g., BGRA format).
        /// </summary>
        /// <param name="length">The number of pixels to include in the span. If 0, uses everything from the offset to the last pixel.</param>
        /// <param name="offset">The pixel offset from the start of the source memory.</param>
        /// <returns>A read-only span of pixels at the specified offset and length.</returns>
        public ReadOnlySpan<uint> GetReadOnlySpan(int length = 0, int offset = 0) => src.GetSpan(length, offset);

        #endregion

        #region Row span accessors

        /// <summary>
        /// Gets a span of bytes for a specific row in the source memory. The length and offset parameters allow for retrieving a specific portion of the row.
        /// </summary>
        /// <param name="y">The row index.</param>
        /// <param name="length">The number of bytes to include in the span. If 0, uses the pixel data of the row from the offset (the row padding is excluded). A longer explicit length can reach into the padding, up to the row stride.</param>
        /// <param name="offset">The byte offset from the start of the row.</param>
        /// <returns>A span of bytes for the specified row.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when the row index, offset, or length is negative or out of range.</exception>
        public Span<byte> GetRowSpanOfBytes(int y, int length = 0, int offset = 0)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(y);
            ArgumentOutOfRangeException.ThrowIfNegative(length);
            ArgumentOutOfRangeException.ThrowIfNegative(offset);

            if (y >= src.Size.Height)
                throw new ArgumentOutOfRangeException(nameof(y), y, $"Row index must be less than the bitmap height ({src.Size.Height}).");

            var rowDataBytes = src.Size.Width * src.BytesPerPixel;
            var maxLength = (length == 0 ? rowDataBytes : src.RowBytes) - offset;

            if (maxLength < 0)
                throw new ArgumentOutOfRangeException(nameof(offset), offset, "Offset value overflow this Bitmap row size.");

            if (length == 0)
            {
                length = maxLength;
            }
            else if (length > maxLength)
            {
                throw new ArgumentOutOfRangeException(nameof(length), length, $"The maximum size allowed for this Bitmap row with an offset of {offset} is {maxLength}.");
            }

            unsafe
            {
                return new(IntPtr.Add(src.Address, checked(y * src.RowBytes + offset)).ToPointer(), length);
            }
        }

        /// <summary>
        /// Gets a read-only span of bytes for a specific row in the source memory.
        /// </summary>
        /// <param name="y">The row index.</param>
        /// <param name="length">The number of bytes to include in the span. If 0, uses the pixel data of the row from the offset (the row padding is excluded).</param>
        /// <param name="offset">The byte offset from the start of the row.</param>
        /// <returns>A read-only span of bytes for the specified row.</returns>
        public ReadOnlySpan<byte> GetReadOnlyRowSpanOfBytes(int y, int length = 0, int offset = 0) => src.GetRowSpanOfBytes(y, length, offset);

        /// <summary>
        /// Gets a span of pixels for a specific row in the source memory, where each pixel is represented as a 32-bit unsigned integer (e.g., BGRA format). The length and offset parameters allow for retrieving a specific portion of the row.
        /// </summary>
        /// <param name="y">The row index.</param>
        /// <param name="length">The number of pixels to include in the span. If 0, uses the width of the row from the offset.</param>
        /// <param name="offset">The pixel offset from the start of the row.</param>
        /// <returns>A span of pixels for the specified row.</returns>
        /// <exception cref="NotSupportedException">Thrown when the framebuffer pixels are not 32-bit.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when the row index, offset, or length is negative or out of range.</exception>
        public Span<uint> GetRowSpan(int y, int length = 0, int offset = 0)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(y);
            ArgumentOutOfRangeException.ThrowIfNegative(length);
            ArgumentOutOfRangeException.ThrowIfNegative(offset);

            if (src.BytesPerPixel != sizeof(uint))
                throw new NotSupportedException($"GetRowSpan<uint> requires a 32-bit pixel format, but the framebuffer uses {src.Format.BitsPerPixel}-bit pixels.");
            if (y >= src.Size.Height)
                throw new ArgumentOutOfRangeException(nameof(y), y, $"Row index must be less than the bitmap height ({src.Size.Height}).");

            var maxLength = src.Size.Width - offset;

            if (maxLength < 0)
                throw new ArgumentOutOfRangeException(nameof(offset), offset, "Offset value overflow this Bitmap row size.");

            if (length == 0)
            {
                length = maxLength;
            }
            else if (length > maxLength)
            {
                throw new ArgumentOutOfRangeException(nameof(length), length, $"The maximum size allowed for this Bitmap row with an offset of {offset} is {maxLength}.");
            }

            unsafe
            {
                return new(IntPtr.Add(src.Address, checked(y * src.RowBytes + offset * sizeof(uint))).ToPointer(), length);
            }
        }

        /// <summary>
        /// Gets a read-only span of pixels for a specific row in the source memory, where each pixel is represented as a 32-bit unsigned integer (e.g., BGRA format).
        /// </summary>
        /// <param name="y">The row index.</param>
        /// <param name="length">The number of pixels to include in the span. If 0, uses the width of the row from the offset.</param>
        /// <param name="offset">The pixel offset from the start of the row.</param>
        /// <returns>A read-only span of pixels for the specified row.</returns>
        public ReadOnlySpan<uint> GetReadOnlyRowSpan(int y, int length = 0, int offset = 0) => src.GetRowSpan(y, length, offset);

        #endregion

        #region 2D span accessors

        /// <summary>
        /// Gets a 2D span of bytes over the whole bitmap, with one row per outer element. Per-row padding is excluded via the span pitch.
        /// </summary>
        /// <returns>A 2D span of bytes representing the entire bitmap.</returns>
        public Span2D<byte> GetSpan2DOfBytes()
        {
            var rowBytes = src.Size.Width * src.BytesPerPixel;
            unsafe
            {
                return new(src.Address.ToPointer(), src.Size.Height, rowBytes, src.RowBytes - rowBytes);
            }
        }

        /// <summary>
        /// Gets a read-only 2D span of bytes over the whole bitmap, with one row per outer element. Per-row padding is excluded via the span pitch.
        /// </summary>
        /// <returns>A read-only 2D span of bytes representing the entire bitmap.</returns>
        public ReadOnlySpan2D<byte> GetReadOnlySpan2DOfBytes() => src.GetSpan2DOfBytes();

        /// <summary>
        /// Gets a 2D span of pixels over the whole bitmap, where each pixel is represented as a 32-bit unsigned integer (e.g., BGRA format). Per-row padding is excluded via the span pitch.
        /// </summary>
        /// <returns>A 2D span of pixels representing the entire bitmap.</returns>
        /// <exception cref="NotSupportedException">Thrown when the framebuffer pixels are not 32-bit.</exception>
        public Span2D<uint> GetSpan2D()
        {
            if (src.BytesPerPixel != sizeof(uint))
                throw new NotSupportedException($"GetSpan2D<uint> requires a 32-bit pixel format, but the framebuffer uses {src.Format.BitsPerPixel}-bit pixels.");

            unsafe
            {
                return new(src.Address.ToPointer(), src.Size.Height, src.Size.Width, src.RowBytes / sizeof(uint) - src.Size.Width);
            }
        }

        /// <summary>
        /// Gets a read-only 2D span of pixels over the whole bitmap, where each pixel is represented as a 32-bit unsigned integer (e.g., BGRA format). Per-row padding is excluded via the span pitch.
        /// </summary>
        /// <returns>A read-only 2D span of pixels representing the entire bitmap.</returns>
        /// <exception cref="NotSupportedException">Thrown when the framebuffer pixels are not 32-bit.</exception>
        public ReadOnlySpan2D<uint> GetReadOnlySpan2D() => src.GetSpan2D();

        /// <summary>
        /// Gets a 2D span of pixels within a region of interest, where each pixel is represented as a 32-bit unsigned integer (e.g., BGRA format).
        /// </summary>
        /// <param name="roi">The region of interest in pixel coordinates.</param>
        /// <returns>A 2D span of pixels representing the region of interest.</returns>
        /// <exception cref="NotSupportedException">Thrown when the framebuffer pixels are not 32-bit.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when the region of interest (ROI) is out of bounds.</exception>
        public Span2D<uint> GetSpan2D(Rectangle roi)
        {
            if (src.BytesPerPixel != sizeof(uint))
                throw new NotSupportedException($"GetSpan2D<uint> requires a 32-bit pixel format, but the framebuffer uses {src.Format.BitsPerPixel}-bit pixels.");
            ValidateRoi(src, roi);

            var pitch = src.RowBytes / sizeof(uint) - roi.Width;

            unsafe
            {
                return new(IntPtr.Add(src.Address, roi.Y * src.RowBytes + roi.X * sizeof(uint)).ToPointer(), roi.Height, roi.Width, pitch);
            }
        }

        /// <summary>
        /// Gets a 2D span of pixels within a region of interest, where each pixel is represented as a 32-bit unsigned integer (e.g., BGRA format).
        /// </summary>
        /// <param name="roi">The region of interest in pixel coordinates.</param>
        /// <returns>A 2D span of pixels representing the region of interest.</returns>
        /// <exception cref="NotSupportedException">Thrown when the framebuffer pixels are not 32-bit.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when the region of interest (ROI) is out of bounds.</exception>
        public Span2D<uint> GetSpan2D(PixelRect roi) => src.GetSpan2D(ToRectangle(roi));

        /// <summary>
        /// Gets a read-only 2D span of pixels within a region of interest, where each pixel is represented as a 32-bit unsigned integer (e.g., BGRA format).
        /// </summary>
        /// <param name="roi">The region of interest in pixel coordinates.</param>
        /// <returns>A read-only 2D span of pixels representing the region of interest.</returns>
        public ReadOnlySpan2D<uint> GetReadOnlySpan2D(Rectangle roi) => src.GetSpan2D(roi);

        /// <summary>
        /// Gets a read-only 2D span of pixels within a region of interest, where each pixel is represented as a 32-bit unsigned integer (e.g., BGRA format).
        /// </summary>
        /// <param name="roi">The region of interest in pixel coordinates.</param>
        /// <returns>A read-only 2D span of pixels representing the region of interest.</returns>
        public ReadOnlySpan2D<uint> GetReadOnlySpan2D(PixelRect roi) => src.GetSpan2D(ToRectangle(roi));

        /// <summary>
        /// Gets a 2D span of bytes within a region of interest. The ROI is specified in pixel coordinates; the span width accounts for the bytes per pixel.
        /// </summary>
        /// <param name="roi">The region of interest in pixel coordinates.</param>
        /// <returns>A 2D span of bytes representing the region of interest.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when the region of interest (ROI) is out of bounds.</exception>
        public Span2D<byte> GetSpan2DOfBytes(Rectangle roi)
        {
            ValidateRoi(src, roi);

            var bytesPerPixel = src.BytesPerPixel;
            var roiBytes = roi.Width * bytesPerPixel;
            var pitch = src.RowBytes - roiBytes;

            unsafe
            {
                return new(IntPtr.Add(src.Address, roi.Y * src.RowBytes + roi.X * bytesPerPixel).ToPointer(), roi.Height, roiBytes, pitch);
            }
        }

        /// <summary>
        /// Gets a 2D span of bytes within a region of interest. The ROI is specified in pixel coordinates; the span width accounts for the bytes per pixel.
        /// </summary>
        /// <param name="roi">The region of interest in pixel coordinates.</param>
        /// <returns>A 2D span of bytes representing the region of interest.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when the region of interest (ROI) is out of bounds.</exception>
        public Span2D<byte> GetSpan2DOfBytes(PixelRect roi) => src.GetSpan2DOfBytes(ToRectangle(roi));

        /// <summary>
        /// Gets a read-only 2D span of bytes within a region of interest. The ROI is specified in pixel coordinates; the span width accounts for the bytes per pixel.
        /// </summary>
        /// <param name="roi">The region of interest in pixel coordinates.</param>
        /// <returns>A read-only 2D span of bytes representing the region of interest.</returns>
        public ReadOnlySpan2D<byte> GetReadOnlySpan2DOfBytes(Rectangle roi) => src.GetSpan2DOfBytes(roi);

        /// <summary>
        /// Gets a read-only 2D span of bytes within a region of interest. The ROI is specified in pixel coordinates; the span width accounts for the bytes per pixel.
        /// </summary>
        /// <param name="roi">The region of interest in pixel coordinates.</param>
        /// <returns>A read-only 2D span of bytes representing the region of interest.</returns>
        public ReadOnlySpan2D<byte> GetReadOnlySpan2DOfBytes(PixelRect roi) => src.GetSpan2DOfBytes(ToRectangle(roi));

        #endregion
    }

    extension(WriteableBitmap src)
    {
        /// <summary>
        /// Gets a <see cref="BitmapInfo"/> struct with the layout of the <see cref="WriteableBitmap"/> (size, stride, bytes per pixel and continuity).
        /// </summary>
        /// <remarks>
        /// The bitmap memory is only valid while it is locked, so the <see cref="BitmapInfo.Address"/> of the returned value is always <see cref="IntPtr.Zero"/>.
        /// Use <see cref="WithBitmapInfo(WriteableBitmap, Action{BitmapInfo})"/> to access the address while the bitmap is locked.
        /// </remarks>
        /// <returns>A <see cref="BitmapInfo"/> struct with information about the bitmap layout.</returns>
        public BitmapInfo GetBitmapInfo()
        {
            using var framebuffer = src.Lock();
            return CreateBitmapInfo(framebuffer, false);
        }

        /// <summary>
        /// Locks the <see cref="WriteableBitmap"/> and runs <paramref name="action"/> with a <see cref="BitmapInfo"/> that includes the memory address.
        /// The address is only valid inside the callback, do not keep it.
        /// </summary>
        /// <param name="action">The action to run while the bitmap is locked.</param>
        public void WithBitmapInfo(Action<BitmapInfo> action)
        {
            ArgumentNullException.ThrowIfNull(action);
            using var framebuffer = src.Lock();
            action(CreateBitmapInfo(framebuffer, true));
        }

        /// <summary>
        /// Locks the <see cref="WriteableBitmap"/> and runs <paramref name="func"/> with a <see cref="BitmapInfo"/> that includes the memory address.
        /// The address is only valid inside the callback, do not keep it.
        /// </summary>
        /// <typeparam name="TResult">The result type.</typeparam>
        /// <param name="func">The function to run while the bitmap is locked.</param>
        /// <returns>The result of <paramref name="func"/>.</returns>
        public TResult WithBitmapInfo<TResult>(Func<BitmapInfo, TResult> func)
        {
            ArgumentNullException.ThrowIfNull(func);
            using var framebuffer = src.Lock();
            return func(CreateBitmapInfo(framebuffer, true));
        }

        /// <summary>
        /// Copies the bitmap pixels into a new <see cref="Mat"/> owned by the caller.
        /// </summary>
        /// <remarks>The channels are in the order of the bitmap pixel format (for <c>Bgra8888</c> it is the OpenCV BGRA order). Premultiplied alpha is not converted.</remarks>
        /// <returns>A new continuous 8-bit Mat with 1, 3 or 4 channels, dispose it when done.</returns>
        /// <exception cref="NotSupportedException">The pixel format is not 8 bits per channel with 1, 3 or 4 channels.</exception>
        public Mat ToMat()
        {
            using var framebuffer = src.Lock();
            using var view = framebuffer.ToMat();
            return view.Clone();
        }
    }
}
