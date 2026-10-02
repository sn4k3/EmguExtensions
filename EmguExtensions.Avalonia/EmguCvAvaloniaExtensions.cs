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

using Avalonia;
using Avalonia.Media.Imaging;
using Avalonia.Platform;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace EmguExtensions.Avalonia;

/// <summary>
/// Extension methods for EmguCV to Avalonia.
/// </summary>
public static class EmguCvAvaloniaExtensions
{
    /// <summary>
    /// Validates the source and converts it, when needed, to an 8-bit Mat. The caller owns (and must dispose) the returned Mat when it is not the same instance as <paramref name="src"/>.
    /// </summary>
    private static Mat PrepareSource(Mat src, Type? srcType, bool applyScale, double scale, double shift)
    {
        ArgumentNullException.ThrowIfNull(src);

        if (src.IsEmpty || src.Width <= 0 || src.Height <= 0)
            throw new ArgumentException("Cannot convert an empty Mat to a WriteableBitmap.", nameof(src));

        var channels = src.NumberOfChannels;
        if (channels is not (1 or 3 or 4))
            throw new NotSupportedException($"Unsupported number of channels: {channels}. Only 1, 3 and 4 channels are supported.");

        if (srcType is not null)
            ValidateColorType(srcType, channels);

        if (src.Depth == DepthType.Cv8U && !applyScale)
            return src;

        if (src.Depth != DepthType.Cv8U && !applyScale)
            throw new NotSupportedException(
                $"Only 8-bit (Cv8U) Mats are supported, got {src.Depth}. Use ToBitmap(scale, shift) to scale other depths to 8-bit.");

        if (!double.IsFinite(scale) || !double.IsFinite(shift))
            throw new ArgumentOutOfRangeException(nameof(scale), "Scale and shift must be finite values.");

        var converted = new Mat();
        try
        {
            src.ConvertTo(converted, DepthType.Cv8U, scale, shift);
            return converted;
        }
        catch
        {
            converted.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Validates that <paramref name="srcType"/> is an Emgu color with as many channels as the Mat.
    /// </summary>
    private static void ValidateColorType(Type srcType, int channels)
    {
        if (!typeof(IColor).IsAssignableFrom(srcType) || !srcType.IsValueType)
            throw new ArgumentException(
                $"The type {srcType.FullName} must be an Emgu.CV.IColor struct (e.g. Gray, Bgr, Bgra, Hsv).", nameof(srcType));

        var dimension = ((IColor)Activator.CreateInstance(srcType)!).Dimension;
        if (dimension != channels)
            throw new ArgumentException(
                $"The color type {srcType.Name} has {dimension} channel(s) but the Mat has {channels}.", nameof(srcType));
    }

    /// <summary>
    /// Creates the bitmap and converts <paramref name="src"/> (8-bit, 1, 3 or 4 channels) straight into the locked
    /// framebuffer memory, without any intermediate Mat.
    /// </summary>
    private static WriteableBitmap CreateBitmap(Mat src, Type? srcType, Vector dpi)
    {
        if (dpi == default)
            dpi = new Vector(96, 96);

        var bitmap = new WriteableBitmap(new PixelSize(src.Width, src.Height), dpi, PixelFormat.Bgra8888,
            AlphaFormat.Unpremul);

        try
        {
            using var framebuffer = bitmap.Lock();
            // View over the framebuffer memory (it honors the row stride), it must be disposed before the lock
            using var target = framebuffer.ToMat();

            switch (src.NumberOfChannels)
            {
                case 1 when srcType is null || srcType == typeof(Gray):
                    CvInvoke.CvtColor(src, target, ColorConversion.Gray2Bgra);
                    break;
                case 3 when srcType is null || srcType == typeof(Bgr):
                    CvInvoke.CvtColor(src, target, ColorConversion.Bgr2Bgra);
                    break;
                case 4 when srcType is null || srcType == typeof(Bgra):
                    src.CopyTo(target);
                    break;
                default:
                    CvInvoke.CvtColor(src, target, srcType!, typeof(Bgra));
                    break;
            }
        }
        catch
        {
            bitmap.Dispose();
            throw;
        }

        return bitmap;
    }

    private static WriteableBitmap ToBitmapCore(Mat src, Type? srcType, bool applyScale, double scale, double shift,
        Vector dpi)
    {
        var prepared = PrepareSource(src, srcType, applyScale, scale, shift);
        try
        {
            return CreateBitmap(prepared, srcType, dpi);
        }
        finally
        {
            if (!ReferenceEquals(prepared, src))
                prepared.Dispose();
        }
    }

    extension(Mat src)
    {
        /// <summary>
        /// Converts the Mat to an Avalonia WriteableBitmap.
        /// The pixels are converted straight into the bitmap memory, without intermediate copies.
        /// </summary>
        /// <param name="srcType">The Emgu <see cref="Emgu.CV.IColor"/> struct type describing the source color space (e.g. <c>typeof(Gray)</c>, <c>typeof(Bgr)</c>, <c>typeof(Bgra)</c>, <c>typeof(Rgba)</c>, <c>typeof(Hsv)</c>). It must have as many channels as the Mat. Used as the source argument to <see cref="CvInvoke.CvtColor(Emgu.CV.IInputArray, Emgu.CV.IOutputArray, System.Type, System.Type)"/> when converting to BGRA.</param>
        /// <param name="dpi">The resolution of the resulting bitmap in dots per inch. Defaults to 96x96 when omitted.</param>
        /// <exception cref="ArgumentException">The Mat is empty, or <paramref name="srcType"/> is not a color matching the Mat channels.</exception>
        /// <exception cref="NotSupportedException">The Mat is not 8-bit or does not have 1, 3 or 4 channels.</exception>
        public WriteableBitmap ToBitmap(Type srcType, Vector dpi = default)
        {
            ArgumentNullException.ThrowIfNull(srcType);
            return ToBitmapCore(src, srcType, false, 1, 0, dpi);
        }

        /// <summary>
        /// Converts the Mat to an Avalonia WriteableBitmap asynchronously.
        /// The pixels are converted straight into the bitmap memory, without intermediate copies.
        /// </summary>
        /// <param name="srcType">The Emgu <see cref="Emgu.CV.IColor"/> struct type describing the source color space (e.g. <c>typeof(Gray)</c>, <c>typeof(Bgr)</c>, <c>typeof(Bgra)</c>, <c>typeof(Rgba)</c>, <c>typeof(Hsv)</c>). It must have as many channels as the Mat.</param>
        /// <param name="dpi">The resolution of the resulting bitmap in dots per inch. Defaults to 96x96 when omitted.</param>
        /// <param name="cancellationToken">The cancellation token, only observed before the conversion starts.</param>
        /// <remarks>The caller must keep the Mat alive and unmodified until the returned task completes.</remarks>
        public Task<WriteableBitmap> ToBitmapAsync(Type srcType, Vector dpi = default,
            CancellationToken cancellationToken = default)
        {
            ArgumentNullException.ThrowIfNull(srcType);
            return Task.Run(() => src.ToBitmap(srcType, dpi), cancellationToken);
        }

        /// <summary>
        /// Converts the Mat to an Avalonia WriteableBitmap.
        /// The pixels are converted straight into the bitmap memory, without intermediate copies.
        /// </summary>
        /// <param name="dpi">The resolution of the resulting bitmap in dots per inch. Defaults to 96x96 when omitted.</param>
        /// <remarks>The Mat is expected to be Gray (1 channel), BGR (3 channels) or BGRA (4 channels).</remarks>
        /// <exception cref="ArgumentException">The Mat is empty.</exception>
        /// <exception cref="NotSupportedException">The Mat is not 8-bit (see the <c>ToBitmap(scale, shift)</c> overload) or does not have 1, 3 or 4 channels.</exception>
        public WriteableBitmap ToBitmap(Vector dpi = default)
        {
            return ToBitmapCore(src, null, false, 1, 0, dpi);
        }

        /// <summary>
        /// Converts the Mat to an Avalonia WriteableBitmap asynchronously.
        /// The pixels are converted straight into the bitmap memory, without intermediate copies.
        /// </summary>
        /// <param name="dpi">The resolution of the resulting bitmap in dots per inch. Defaults to 96x96 when omitted.</param>
        /// <param name="cancellationToken">The cancellation token, only observed before the conversion starts.</param>
        /// <remarks>The caller must keep the Mat alive and unmodified until the returned task completes.</remarks>
        public Task<WriteableBitmap> ToBitmapAsync(Vector dpi = default, CancellationToken cancellationToken = default)
        {
            return Task.Run(() => src.ToBitmap(dpi), cancellationToken);
        }

        /// <summary>
        /// Converts a Mat of any depth to an Avalonia WriteableBitmap, scaling it to 8-bit first with <c>dst = saturate(src * scale + shift)</c>.
        /// </summary>
        /// <param name="scale">The factor applied to every value, e.g. <c>255.0 / 65535</c> for 16-bit images or <c>255</c> for floating point images in the 0..1 range.</param>
        /// <param name="shift">The value added after scaling.</param>
        /// <param name="dpi">The resolution of the resulting bitmap in dots per inch. Defaults to 96x96 when omitted.</param>
        /// <remarks>The Mat is expected to be Gray (1 channel), BGR (3 channels) or BGRA (4 channels). An 8-bit scaled copy is created, the source is not modified.</remarks>
        /// <exception cref="ArgumentException">The Mat is empty.</exception>
        /// <exception cref="NotSupportedException">The Mat does not have 1, 3 or 4 channels.</exception>
        public WriteableBitmap ToBitmap(double scale, double shift = 0, Vector dpi = default)
        {
            return ToBitmapCore(src, null, true, scale, shift, dpi);
        }

        /// <summary>
        /// Converts a Mat of any depth to an Avalonia WriteableBitmap asynchronously, scaling it to 8-bit first with <c>dst = saturate(src * scale + shift)</c>.
        /// </summary>
        /// <param name="scale">The factor applied to every value, e.g. <c>255.0 / 65535</c> for 16-bit images or <c>255</c> for floating point images in the 0..1 range.</param>
        /// <param name="shift">The value added after scaling.</param>
        /// <param name="dpi">The resolution of the resulting bitmap in dots per inch. Defaults to 96x96 when omitted.</param>
        /// <param name="cancellationToken">The cancellation token, only observed before the conversion starts.</param>
        /// <remarks>The caller must keep the Mat alive and unmodified until the returned task completes.</remarks>
        public Task<WriteableBitmap> ToBitmapAsync(double scale, double shift = 0, Vector dpi = default,
            CancellationToken cancellationToken = default)
        {
            return Task.Run(() => src.ToBitmap(scale, shift, dpi), cancellationToken);
        }
    }
}
