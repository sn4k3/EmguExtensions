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
    extension(Mat src)
    {
        /// <summary>
        /// Converts the Mat to an Avalonia WriteableBitmap.
        /// Optimized to avoid intermediate copies where possible using direct memory locking.
        /// </summary>
        /// <param name="srcType">The Emgu <see cref="Emgu.CV.IColor"/> struct type describing the source color space (e.g. <c>typeof(Gray)</c>, <c>typeof(Bgr)</c>, <c>typeof(Hsv)</c>). Used as the source argument to <see cref="CvInvoke.CvtColor(Emgu.CV.IInputArray, Emgu.CV.IOutputArray, System.Type, System.Type)"/> when converting to BGRA.</param>
        /// <param name="dpi">The resolution of the resulting bitmap in dots per inch. Defaults to 96x96 when omitted.</param>
        public WriteableBitmap ToBitmap(Type srcType, Vector dpi = default)
        {
            if (src.Depth != DepthType.Cv8U)
                throw new NotSupportedException($"Only 8-bit (Cv8U) Mats are supported, got {src.Depth}.");

            if (dpi == default) dpi = new Vector(96, 96);
            var writableBitmap = new WriteableBitmap(new PixelSize(src.Width, src.Height), dpi, PixelFormat.Bgra8888, AlphaFormat.Unpremul);
            if (src.IsEmpty) return writableBitmap;

            try
            {
                using var lockBuffer = writableBitmap.Lock();

                switch (src.NumberOfChannels)
                {
                    case 1:
                    case 3:
                    {
                        using var convertMat = new Mat();
                        CvInvoke.CvtColor(src, convertMat, srcType, typeof(Bgra));
                        convertMat.CopyTo(lockBuffer.Address);
                        break;
                    }
                    case 4:
                    {
                        src.CopyTo(lockBuffer.Address);
                        break;
                    }
                    default:
                        throw new NotSupportedException($"Unsupported number of channels: {src.NumberOfChannels}.");
                }
            }
            catch
            {
                writableBitmap.Dispose();
                throw;
            }

            return writableBitmap;
        }


        /// <summary>
        /// Converts the Mat to an Avalonia WriteableBitmap asynchronously.
        /// Optimized to avoid intermediate copies where possible using direct memory locking.
        /// </summary>
        /// <param name="srcType">The Emgu <see cref="Emgu.CV.IColor"/> struct type describing the source color space (e.g. <c>typeof(Gray)</c>, <c>typeof(Bgr)</c>, <c>typeof(Hsv)</c>). Used as the source argument to <see cref="CvInvoke.CvtColor(Emgu.CV.IInputArray, Emgu.CV.IOutputArray, System.Type, System.Type)"/> when converting to BGRA.</param>
        /// <param name="dpi">The resolution of the resulting bitmap in dots per inch. Defaults to 96x96 when omitted.</param>
        public Task<WriteableBitmap> ToBitmapAsync(Type srcType, Vector dpi = default)
        {
            return Task.Run(() => src.ToBitmap(srcType, dpi));
        }

        /// <summary>
        /// Converts the Mat to an Avalonia WriteableBitmap.
        /// Optimized to avoid intermediate copies where possible using direct memory locking.
        /// </summary>
        /// <param name="dpi">The resolution of the resulting bitmap in dots per inch. Defaults to 96x96 when omitted.</param>
        public WriteableBitmap ToBitmap(Vector dpi = default)
        {
            if (src.Depth != DepthType.Cv8U)
                throw new NotSupportedException($"Only 8-bit (Cv8U) Mats are supported, got {src.Depth}.");

            if (dpi == default) dpi = new Vector(96, 96);
            var writableBitmap = new WriteableBitmap(new PixelSize(src.Width, src.Height), dpi, PixelFormat.Bgra8888, AlphaFormat.Unpremul);
            if (src.IsEmpty) return writableBitmap;

            try
            {
                using var lockBuffer = writableBitmap.Lock();

                switch (src.NumberOfChannels)
                {
                    case 1:
                    {
                        using var convertMat = new Mat();
                        CvInvoke.CvtColor(src, convertMat, ColorConversion.Gray2Bgra);
                        convertMat.CopyTo(lockBuffer.Address);
                        break;
                    }
                    case 3:
                    {
                        using var convertMat = new Mat();
                        CvInvoke.CvtColor(src, convertMat, ColorConversion.Bgr2Bgra);
                        convertMat.CopyTo(lockBuffer.Address);
                        break;
                    }
                    case 4:
                    {
                        src.CopyTo(lockBuffer.Address);
                        break;
                    }
                    default:
                        throw new NotSupportedException($"Unsupported number of channels: {src.NumberOfChannels}.");
                }
            }
            catch
            {
                writableBitmap.Dispose();
                throw;
            }

            return writableBitmap;
        }

        /// <summary>
        /// Converts the Mat to an Avalonia WriteableBitmap asynchronously.
        /// Optimized to avoid intermediate copies where possible using direct memory locking.
        /// </summary>
        /// <param name="dpi">The resolution of the resulting bitmap in dots per inch. Defaults to 96x96 when omitted.</param>
        public Task<WriteableBitmap> ToBitmapAsync(Vector dpi = default)
        {
            return Task.Run(() => src.ToBitmap(dpi));
        }
    }
}