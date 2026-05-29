# EmguExtensions.Avalonia

[![License](https://img.shields.io/github/license/sn4k3/EmguExtensions?style=for-the-badge)](https://github.com/sn4k3/EmguExtensions/blob/main/LICENSE)
[![GitHub repo size](https://img.shields.io/github/repo-size/sn4k3/EmguExtensions?style=for-the-badge)](#)
[![Code size](https://img.shields.io/github/languages/code-size/sn4k3/EmguExtensions?style=for-the-badge)](#)
[![NuGet](https://img.shields.io/nuget/v/EmguExtensions.Avalonia?style=for-the-badge)](https://www.nuget.org/packages/EmguExtensions.Avalonia)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/sn4k3?color=red&style=for-the-badge)](https://github.com/sponsors/sn4k3)

Avalonia integration for [EmguExtensions](https://www.nuget.org/packages/EmguExtensions). It converts `Emgu.CV.Mat` images into Avalonia `WriteableBitmap` instances and exposes span-based access to locked Avalonia framebuffers.

## Features

- Convert `Mat` to `WriteableBitmap`
- Convert grayscale, BGR, and BGRA 8-bit Mats to Avalonia `Bgra8888`
- Optional source color type conversion via Emgu.CV color structs
- Async conversion helpers for background image preparation
- Span and `Span2D` access over `ILockedFramebuffer`
- Row-aware access that handles framebuffer stride/padding
- Bitmap metadata helper via `GetBitmapInfo()`

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

The `srcType` is passed to Emgu.CV color conversion and converted to `Bgra`.

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

`ToBitmapAsync()` uses `Task.Run`. Assign Avalonia UI properties on the UI thread.

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

### ROI Spans

```csharp
using System.Drawing;
using EmguExtensions.Avalonia;

using var framebuffer = bitmap.Lock();

var roi = new Rectangle(10, 10, 100, 80);

Span2D<byte> roiBytes = framebuffer.GetSpan2DOfBytes(roi);
Span2D<uint> roiPixels = framebuffer.GetSpan2D(roi);
```

## Bitmap Info

```csharp
using EmguExtensions.Avalonia;

BitmapInfo info = bitmap.GetBitmapInfo();

Console.WriteLine($"{info.Width}x{info.Height}, row bytes: {info.RowBytes}");
```

## Limitations

- Only `DepthType.Cv8U` Mats are supported.
- `ToBitmap()` supports 1, 3, and 4 channel Mats.
- Output bitmap format is always `PixelFormat.Bgra8888` with `AlphaFormat.Unpremul`.
- A 4-channel Mat is copied directly, so it should already be BGRA-compatible.
- Framebuffer spans are valid only while the framebuffer lock is alive.
- Flat `Span<uint>` access requires a 32-bit continuous framebuffer. Use 2D spans for padded rows.

## License

MIT. See the repository [LICENSE](https://github.com/sn4k3/EmguExtensions/blob/master/LICENSE).
