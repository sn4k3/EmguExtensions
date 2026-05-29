using System.Drawing;
using Emgu.CV;

namespace EmguExtensions;

/// <summary>
/// Struct to hold information about a bitmap, including its dimensions, memory address, and whether the memory is contiguous.
/// </summary>
public readonly record struct BitmapInfo
{
    /// <summary>
    /// Gets the memory address.
    /// </summary>
    public nint Address { get; init; } = IntPtr.Zero;

    /// <summary>
    /// Gets a value indicating whether the bitmap's memory is contiguous.
    /// </summary>
    public bool IsContiguous { get; init; } = true;

    /// <summary>
    /// Gets the number of bytes per pixel.
    /// </summary>
    /// <remarks>1 = Greyscale, 3 = BGR, 4 = BGRA</remarks>
    public int BytesPerPixel { get; init; } = 1;

    /// <summary>
    /// Gets the total number of bytes in a single row.
    /// </summary>
    public int RowBytes { get; init; }

    /// <summary>
    /// Gets the width of the bitmap.
    /// </summary>
    public int Width { get; init; } = 0;

    /// <summary>
    /// Gets the height of the bitmap.
    /// </summary>

    public int Height { get; init; } = 0;

    /// <summary>
    /// Gets the size of the bitmap as a <see cref="Size"/> struct, which contains the width and height.
    /// </summary>
    public Size Size
    {
        get => new(Width, Height);
        init { Width = value.Width; Height = value.Height; }
    }

    /// <summary>
    /// Gets the total number of pixels.
    /// </summary>
    public int PixelCount => Width * Height;

    /// <summary>
    /// Initializes a new instance of the <see cref="BitmapInfo"/> struct with default values. The memory address is set to zero, dimensions are set to zero, element size is set to one, and the memory is assumed to be contiguous.
    /// </summary>
    public BitmapInfo()
    {
    }
}