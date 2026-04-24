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

using System;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;

namespace EmguExtensions;

/// <summary>
/// Represents a region of interest (ROI) within a source matrix, providing access to a cropped area and related
/// metadata.
/// </summary>
/// <remarks>The MatRoi class enables cropping and manipulation of a specified rectangular region within a source
/// Mat object. It supports optional padding around the ROI and allows callers to control whether the source Mat is
/// disposed when the MatRoi instance is disposed, via the LeaveOpen property. This class is useful for scenarios where
/// working with subregions of an image or matrix is required, such as image analysis or preprocessing tasks.</remarks>
public class MatRoi : LeaveOpenDisposableObject
{
    /// <summary>
    /// Gets the source matrix used for processing.
    /// </summary>
    public Mat SourceMat { get; }

    /// <summary>
    /// Gets the cropped matrix representing the region of interest (ROI) defined by the specified rectangle
    /// </summary>
    public Mat RoiMat { get; }

    /// <summary>
    /// Gets the bounding rectangle that defines the area occupied by the object.
    /// </summary>
    public Rectangle Roi { get; }

    /// <summary>
    /// Gets if the <see cref="SourceMat"/> is the same size of the <see cref="Roi"/>
    /// </summary>
    public bool IsSourceSameSizeOfRoi => SourceMat.Size == Roi.Size;

    /// <summary>
    /// Initializes a new instance of the MatRoi class, representing a region of interest (ROI) within a source matrix.
    /// </summary>
    /// <remarks>The RoiMat property contains the cropped region defined by the roi parameter. Ensure that
    /// the roi parameter does not exceed the dimensions of the source matrix to avoid runtime errors.</remarks>
    /// <param name="mat">The source matrix from which the region of interest is derived. This matrix must not be null.</param>
    /// <param name="roi">The rectangle that defines the region of interest within the source matrix. The rectangle must be fully
    /// contained within the bounds of the source matrix.</param>
    /// <param name="leaveOpen">true to leave the source matrix open after the MatRoi instance is disposed; otherwise, false to dispose the
    /// source matrix when the MatRoi instance is disposed. Defaults to <see langword="true"/> because the caller typically owns the source Mat.</param>
    public MatRoi(Mat mat, Rectangle roi, bool leaveOpen = true) : base(leaveOpen)
    {
        ArgumentNullException.ThrowIfNull(mat);
        SourceMat = mat;
        RoiMat = mat.SafeRoi(roi, out var rectangle);
        Roi = rectangle;
    }

    /// <summary>
    /// Initializes a new instance of the MatRoi class, representing a region of interest (ROI) within a source matrix,
    /// with optional padding on each side.
    /// </summary>
    /// <param name="mat">The source matrix from which the region of interest is extracted. Cannot be null.</param>
    /// <param name="roi">The rectangle that defines the region of interest within the source matrix.</param>
    /// <param name="padLeft">The number of pixels to add as padding to the left side of the ROI. Must be zero or greater.</param>
    /// <param name="padTop">The number of pixels to add as padding to the top side of the ROI. Must be zero or greater.</param>
    /// <param name="padRight">The number of pixels to add as padding to the right side of the ROI. Must be zero or greater.</param>
    /// <param name="padBottom">The number of pixels to add as padding to the bottom side of the ROI. Must be zero or greater.</param>
    /// <param name="leaveOpen">true to leave the source matrix open after the MatRoi is disposed; otherwise, false.
    /// Defaults to <see langword="true"/> because the caller typically owns the source Mat.</param>
    public MatRoi(Mat mat, Rectangle roi, int padLeft, int padTop, int padRight, int padBottom, bool leaveOpen = true) : base(leaveOpen)
    {
        ArgumentNullException.ThrowIfNull(mat);
        SourceMat = mat;
        RoiMat = mat.SafeRoi(roi, out var rectangle, padLeft, padTop, padRight, padBottom);
        Roi = rectangle;
    }

    /// <summary>
    /// Initializes a new instance of the MatRoi class, representing a region of interest (ROI) within a source matrix
    /// with optional padding.
    /// </summary>
    /// <remarks>The cropped matrix is created based on the specified region of interest and padding. Ensure
    /// that the region of interest and padding do not exceed the bounds of the source matrix to avoid runtime
    /// exceptions.</remarks>
    /// <param name="mat">The source matrix from which the region of interest is extracted. This parameter cannot be null.</param>
    /// <param name="roi">The rectangle that defines the region of interest within the source matrix. The rectangle must be fully
    /// contained within the bounds of the source matrix.</param>
    /// <param name="padding">The amount of padding, in pixels, to apply around the region of interest. Padding increases the size of the
    /// cropped region if space allows.</param>
    /// <param name="leaveOpen">true to leave the source matrix open and accessible after the MatRoi instance is disposed; otherwise, false to
    /// dispose the source matrix when the MatRoi is disposed. Defaults to <see langword="true"/> because the caller typically owns the source Mat.</param>
    public MatRoi(Mat mat, Rectangle roi, Size padding, bool leaveOpen = true) : base(leaveOpen)
    {
        ArgumentNullException.ThrowIfNull(mat);
        SourceMat = mat;
        RoiMat = mat.SafeRoi(roi, out var rectangle, padding);
        Roi = rectangle;
    }

    /// <summary>
    /// Initializes a new instance of the MatRoi class, representing a region of interest (ROI) within a source matrix
    /// with optional padding.
    /// </summary>
    /// <remarks>The cropped matrix is created based on the specified region of interest and padding. Ensure
    /// that the region of interest and padding do not exceed the bounds of the source matrix to avoid runtime
    /// exceptions.</remarks>
    /// <param name="mat">The source matrix from which the region of interest is extracted. This parameter cannot be null.</param>
    /// <param name="roi">The rectangle that defines the region of interest within the source matrix. The rectangle must be fully
    /// contained within the bounds of the source matrix.</param>
    /// <param name="padding">The amount of padding, in pixels, to apply around the region of interest. Padding increases the size of the
    /// cropped region if space allows.</param>
    /// <param name="leaveOpen">true to leave the source matrix open and accessible after the MatRoi instance is disposed; otherwise, false to
    /// dispose the source matrix when the MatRoi is disposed. Defaults to <see langword="true"/> because the caller typically owns the source Mat.</param>
    public MatRoi(Mat mat, Rectangle roi, int padding, bool leaveOpen = true) : base(leaveOpen)
    {
        ArgumentNullException.ThrowIfNull(mat);
        SourceMat = mat;
        RoiMat = mat.SafeRoi(roi, out var rectangle, padding);
        Roi = rectangle;
    }

    /// <summary>
    /// Initializes a new instance of the MatRoi class based on an existing MatRoi instance and a new region of interest (ROI) rectangle.
    /// </summary>
    /// <param name="matRoi">The existing MatRoi instance from which to create the new instance. Cannot be null.</param>
    /// <param name="roi">The rectangle that defines the new region of interest within the source matrix. The rectangle must be fully
    /// contained within the bounds of the source matrix.</param>
    public MatRoi(MatRoi matRoi, Rectangle roi) : this(matRoi.SourceMat, roi, true)
    {
    }

    /// <summary>
    /// Initializes a new instance of the MatRoi class by loading an image from the specified file path and defining a region of interest (ROI) within that image.
    /// </summary>
    /// <param name="filePath">The file path of the image to load.</param>
    /// <param name="loadType">The type of image to load, specified by the <see cref="ImreadModes"/> enumeration.</param>
    /// <param name="roi">The rectangle that defines the region of interest within the loaded image.</param>
    /// <param name="padLeft">The number of pixels to add as padding to the left side of the ROI. Must be zero or greater.</param>
    /// <param name="padTop">The number of pixels to add as padding to the top side of the ROI. Must be zero or greater.</param>
    /// <param name="padRight">The number of pixels to add as padding to the right side of the ROI. Must be zero or greater.</param>
    /// <param name="padBottom">The number of pixels to add as padding to the bottom side of the ROI. Must be zero or greater.</param>
    public MatRoi(string filePath, ImreadModes loadType, Rectangle roi, int padLeft = 0, int padTop = 0, int padRight = 0, int padBottom = 0) : this(new Mat(filePath, loadType), roi, padLeft, padTop, padRight, padBottom, false)
    {
    }

    /// <summary>
    /// Initializes a new instance of the MatRoi class by loading an image from the specified file path and defining a region of interest (ROI) within that image.
    /// </summary>
    /// <param name="filePath">The file path of the image to load.</param>
    /// <param name="loadType">The type of image to load, specified by the <see cref="ImreadModes"/> enumeration.</param>
    /// <param name="roi">The rectangle that defines the region of interest within the loaded image.</param>
    /// <param name="padding">The amount of padding to apply around the region of interest. The width and height of the <see cref="Size"/> structure specify the horizontal and vertical padding, respectively.</param>
    public MatRoi(string filePath, ImreadModes loadType, Rectangle roi, Size padding)
        : this(filePath, loadType, roi, padding.Width, padding.Height, padding.Width, padding.Height)
    {
    }

    /// <summary>
    /// Initializes a new instance of the MatRoi class by loading an image from the specified file path and defining a region of interest (ROI) within that image.
    /// </summary>
    /// <param name="filePath">The file path of the image to load.</param>
    /// <param name="loadType">The type of image to load, specified by the <see cref="ImreadModes"/> enumeration.</param>
    /// <param name="roi">The rectangle that defines the region of interest within the loaded image.</param>
    /// <param name="padding">The amount of padding to apply around the region of interest. The width and height of the <see cref="Size"/> structure specify the horizontal and vertical padding, respectively.</param>
    public MatRoi(string filePath, ImreadModes loadType, Rectangle roi, int padding)
        : this(filePath, loadType, roi, padding, padding, padding, padding)
    {
    }

    /// <summary>
    /// Initializes a new instance of the MatRoi class using the specified image file and region of interest.
    /// </summary>
    /// <param name="filePath">The path to the image file to load. Cannot be null or empty.</param>
    /// <param name="roi">The rectangular region of interest to select within the loaded image.</param>
    public MatRoi(string filePath, Rectangle roi) : this(new Mat(filePath), roi, false)
    {
    }

    /// <summary>
    /// Initializes a new instance of the MatRoi class by loading an image from the specified file path and defining a region of interest (ROI) within that image.
    /// </summary>
    /// <param name="filePath">The file path of the image to load.</param>
    /// <param name="roi">The rectangle that defines the region of interest within the loaded image.</param>
    /// <param name="padLeft">The number of pixels to add as padding to the left side of the ROI. Must be zero or greater.</param>
    /// <param name="padTop">The number of pixels to add as padding to the top side of the ROI. Must be zero or greater.</param>
    /// <param name="padRight">The number of pixels to add as padding to the right side of the ROI. Must be zero or greater.</param>
    /// <param name="padBottom">The number of pixels to add as padding to the bottom side of the ROI. Must be zero or greater.</param>
    public MatRoi(string filePath, Rectangle roi, int padLeft = 0, int padTop = 0, int padRight = 0, int padBottom = 0)
        : this(new Mat(filePath), roi, padLeft, padTop, padRight, padBottom, false)
    {
    }

    /// <summary>
    /// Initializes a new instance of the MatRoi class by loading an image from the specified file path and defining a region of interest (ROI) within that image.
    /// </summary>
    /// <param name="filePath">The file path of the image to load.</param>
    /// <param name="roi">The rectangle that defines the region of interest within the loaded image.</param>
    /// <param name="padding">The amount of padding to apply around the region of interest. The width and height of the <see cref="Size"/> structure specify the horizontal and vertical padding, respectively.</param>
    public MatRoi(string filePath, Rectangle roi, Size padding)
        : this(filePath, roi, padding.Width, padding.Height, padding.Width, padding.Height)
    {
    }

    /// <summary>
    /// Initializes a new instance of the MatRoi class by loading an image from the specified file path and defining a region of interest (ROI) within that image.
    /// </summary>
    /// <param name="filePath">The file path of the image to load.</param>
    /// <param name="roi">The rectangle that defines the region of interest within the loaded image.</param>
    /// <param name="padding">The amount of padding to apply around the region of interest. The width and height of the <see cref="Size"/> structure specify the horizontal and vertical padding, respectively.</param>
    public MatRoi(string filePath, Rectangle roi, int padding)
        : this(filePath, roi, padding, padding, padding, padding)
    {
    }

    /// <summary>
    /// Creates a new instance of the MatRoi class by calculating the bounding rectangle of the non-zero pixels in the provided source matrix.
    /// </summary>
    /// <param name="mat">The source matrix from which the region of interest is derived. This matrix must not be null.</param>
    /// <param name="leaveOpen">true to leave the source matrix open after the MatRoi instance is disposed; otherwise, false to dispose the
    /// source matrix when the MatRoi instance is disposed. Defaults to <see langword="true"/> because the caller typically owns the source Mat.</param>
    public static MatRoi CreateFromBoundingRectangle(Mat mat, bool leaveOpen = true)
    {
        if (mat.IsEmpty) return new MatRoi(mat, Rectangle.Empty, leaveOpen);
        var boundingRectangle = CvInvoke.BoundingRectangle(mat);
        return new MatRoi(mat, boundingRectangle, leaveOpen);
    }

    /// <inheritdoc />
    protected override void DisposeManaged()
    {
        RoiMat.Dispose();
        if (!LeaveOpen)
        {
            SourceMat.Dispose();
        }
    }
}