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

using System.ComponentModel;

namespace EmguExtensions;

/// <summary>
/// Specifies the behavior when an empty Region of Interest (ROI) is encountered in image processing operations.
/// </summary>
public enum EmptyRoiBehavior
{
    /// <summary>
    /// Uses OpenCV's default behavior when the requested ROI is empty, typically producing an empty matrix.
    /// </summary>
    [Description("Use OpenCV's default behavior when the requested ROI is empty")]
    Default,

    /// <summary>
    /// Uses the full source image when the requested ROI is empty.
    /// </summary>
    [Description("Use the full source image when the requested ROI is empty")]
    CaptureSource,

    /// <summary>
    /// Throws an <see cref="InvalidOperationException"/> when the requested ROI is empty.
    /// </summary>
    [Description("Throw InvalidOperationException when the requested ROI is empty")]
    ThrowException
}