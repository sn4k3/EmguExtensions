# EmguExtensions.Avalonia

[![License](https://img.shields.io/github/license/sn4k3/EmguExtensions?style=for-the-badge)](https://github.com/sn4k3/EmguExtensions/blob/main/LICENSE)
[![GitHub repo size](https://img.shields.io/github/repo-size/sn4k3/EmguExtensions?style=for-the-badge)](#)
[![Code size](https://img.shields.io/github/languages/code-size/sn4k3/EmguExtensions?style=for-the-badge)](#)
[![NuGet](https://img.shields.io/nuget/v/EmguExtensions.Avalonia?style=for-the-badge)](https://www.nuget.org/packages/EmguExtensions.Avalonia)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/sn4k3?color=red&style=for-the-badge)](https://github.com/sponsors/sn4k3)

Avalonia integration for [EmguExtensions](https://www.nuget.org/packages/EmguExtensions). It converts `Emgu.CV.Mat` images into Avalonia `WriteableBitmap` instances and exposes span-based access to locked Avalonia framebuffers.

## Features

- Convert `Mat` to `WriteableBitmap`
- Convert grayscale, BGR, and BGRA 8-bit Mats to Avalonia `Bgra8888`, converting straight into the bitmap memory (no intermediate Mat)
- Optional source color type conversion via Emgu.CV color structs (`Rgba`, `Hsv`, ...)
- Convert other depths (16-bit, floating point) with a scale/shift to 8-bit
- Async conversion helpers for background image preparation
- Zero-copy `Mat` view over a locked framebuffer (`ToMat()`), and `WriteableBitmap.ToMat()` to get a copy
- Span and `Span2D` access over `ILockedFramebuffer`, with `System.Drawing` and Avalonia `PixelPoint`/`PixelRect` overloads
- Row-aware access that handles framebuffer stride/padding
- Bitmap metadata helper via `GetBitmapInfo()` and safe, lock-scoped address access via `WithBitmapInfo()`

## Requirements

- .NET 10 or later
- Avalonia 12.0.4 or later
- EmguExtensions
- Emgu.CV runtime package matching your target platform

## Installation

### .NET CLI

```bash
dotnet add package EmguExtensions.Avalonia
```

### NuGet Package Manager

```powershell
Install-Package EmguExtensions.Avalonia
```

## Quick Start

### Convert Mat to WriteableBitmap

```csharp
using Avalonia.Media.Imaging;
using Emgu.CV;
using Emgu.CV.CvEnum;
using EmguExtensions.Avalonia;

using var mat = CvInvoke.Imread("image.png", ImreadModes.Color);

WriteableBitmap bitmap = mat.ToBitmap();
PreviewImage.Source = bitmap;
```

`ToBitmap()` supports 8-bit Mats with 1, 3, or 4 channels:

- 1 channel: `Gray -> BGRA`
- 3 channels: `BGR -> BGRA`
- 4 channels: copied directly

The returned bitmap is owned by the caller. Keep it alive while UI uses it, and dispose it when replaced or no longer needed.

### Convert with Explicit Source Color Type

Use this overload when the `Mat` channel count alone is not enough to describe the color space.

```csharp
using Emgu.CV.Structure;
using EmguExtensions.Avalonia;

WriteableBitmap bitmap = hsvMat.ToBitmap(typeof(Hsv));
```

The `srcType` must be an Emgu.CV color struct (`Gray`, `Bgr`, `Bgra`, `Rgba`, `Hsv`, ...) with as many channels as the `Mat`, otherwise an `ArgumentException` is thrown. It is passed to Emgu.CV color conversion and converted to `Bgra`.

### Convert Other Depths

Mats that are not 8-bit are scaled to 8-bit with `saturate(value * scale + shift)`:

```csharp
// 16-bit grayscale: map 0..65535 to 0..255
WriteableBitmap bitmap = mat16.ToBitmap(255.0 / 65535);

// Floating point in the 0..1 range
WriteableBitmap preview = matFloat.ToBitmap(255);
```

The source `Mat` is not modified.

### Convert on Background Thread

```csharp
using Avalonia.Threading;
using EmguExtensions.Avalonia;

var bitmap = await mat.ToBitmapAsync();

await Dispatcher.UIThread.InvokeAsync(() =>
{
    PreviewImage.Source = bitmap;
});
```

`ToBitmapAsync()` uses `Task.Run`: the cancellation token is only observed before the conversion starts, and the `Mat` must stay alive and unmodified until the task completes. Assign Avalonia UI properties on the UI thread.

### Framebuffer to Mat

```csharp
using var framebuffer = bitmap.Lock();
using var view = framebuffer.ToMat(); // zero-copy, honors the row stride

CvInvoke.Rectangle(view, new Rectangle(10, 10, 50, 50), new MCvScalar(0, 0, 255, 255), -1);
// Dispose the view BEFORE the framebuffer
```

The channels follow the framebuffer format (`Bgra8888` is the OpenCV BGRA order). Use `bitmap.ToMat()` to get an independent copy.

## Framebuffer Span Access

Lock a `WriteableBitmap`, then use span helpers on the locked framebuffer.

```csharp
using EmguExtensions.Avalonia;

using var framebuffer = bitmap.Lock();

Span2D<byte> bytes = framebuffer.GetSpan2DOfBytes();

int x = 10;
int y = 20;
int offset = x * framebuffer.BytesPerPixel;

bytes[y, offset + 0] = 255; // B
bytes[y, offset + 1] = 0;   // G
bytes[y, offset + 2] = 0;   // R
bytes[y, offset + 3] = 255; // A
```

For 32-bit framebuffers, pixel spans are also available:

```csharp
using var framebuffer = bitmap.Lock();

Span2D<uint> pixels = framebuffer.GetSpan2D();
pixels[20, 10] = 0xFFFF0000;
```

Use byte spans when exact channel order matters. Avalonia `Bgra8888` stores bytes as B, G, R, A.

## Available Framebuffer Helpers

### Metadata

```csharp
using var framebuffer = bitmap.Lock();

int bytesPerPixel = framebuffer.BytesPerPixel;
int byteCount = framebuffer.ByteCount;
int pixelCount = framebuffer.PixelCount;
bool isContinuous = framebuffer.IsContinuous;
```

### Flat Spans

```csharp
using var framebuffer = bitmap.Lock();

Span<byte> allBytes = framebuffer.GetSpanOfBytes();
ReadOnlySpan<byte> readonlyBytes = framebuffer.GetReadOnlySpanOfBytes();

Span<uint> pixels = framebuffer.GetSpan();
ReadOnlySpan<uint> readonlyPixels = framebuffer.GetReadOnlySpan();
```

Flat pixel spans require:

- 32-bit pixel format
- continuous framebuffer memory

If the framebuffer has row padding, use `GetSpan2D()` or row spans.

### Row Spans

```csharp
using var framebuffer = bitmap.Lock();

Span<byte> rowBytes = framebuffer.GetRowSpanOfBytes(y: 5);
Span<uint> rowPixels = framebuffer.GetRowSpan(y: 5);
```

`GetRowSpanOfBytes` returns the pixel data of the row without the padding; pass an explicit `length` to reach into it (up to the row stride). A `length` of `0` means "everything available", negative values throw.

### ROI Spans

```csharp
using System.Drawing;
using EmguExtensions.Avalonia;

using var framebuffer = bitmap.Lock();

var roi = new Rectangle(10, 10, 100, 80); // or an Avalonia PixelRect

Span2D<byte> roiBytes = framebuffer.GetSpan2DOfBytes(roi);
Span2D<uint> roiPixels = framebuffer.GetSpan2D(roi);
```

## Bitmap Info

```csharp
using EmguExtensions.Avalonia;

BitmapInfo info = bitmap.GetBitmapInfo();

Console.WriteLine($"{info.Width}x{info.Height}, row bytes (stride): {info.RowBytes}");

// The address is only valid while the bitmap is locked, so it is only exposed inside the callback
bitmap.WithBitmapInfo(info =>
{
    nint address = info.Address;
    // ...
});
```

`GetBitmapInfo()` never returns the memory address (`Address` is `IntPtr.Zero`), because the bitmap memory is only guaranteed while it is locked.

## Limitations

- `ToBitmap()` and `ToBitmap(srcType)` only support `DepthType.Cv8U` Mats, use `ToBitmap(scale, shift)` for other depths.
- Mats must have 1, 3, or 4 channels.
- Output bitmap format is always `PixelFormat.Bgra8888` with `AlphaFormat.Unpremul`.
- A 4-channel Mat is copied directly unless a different `srcType` (e.g. `Rgba`) is given, so it should already be BGRA-compatible.
- Framebuffer spans are valid only while the framebuffer lock is alive.
- Flat `Span<uint>` access requires a 32-bit continuous framebuffer. Use 2D spans for padded rows.

## License

MIT. See the repository [LICENSE](https://github.com/sn4k3/EmguExtensions/blob/master/LICENSE).
