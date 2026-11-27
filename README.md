# [![Logo](https://raw.githubusercontent.com/sn4k3/EmguExtensions/main/media/EmguExtensions-32.png)](#) EmguExtensions

[![License](https://img.shields.io/github/license/sn4k3/EmguExtensions?style=for-the-badge)](https://github.com/sn4k3/EmguExtensions/blob/master/LICENSE)
[![GitHub repo size](https://img.shields.io/github/repo-size/sn4k3/EmguExtensions?style=for-the-badge)](#)
[![Code size](https://img.shields.io/github/languages/code-size/sn4k3/EmguExtensions?style=for-the-badge)](#)
[![NuGet](https://img.shields.io/nuget/v/EmguExtensions?style=for-the-badge)](https://www.nuget.org/packages/EmguExtensions)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/sn4k3?color=red&style=for-the-badge)](https://github.com/sponsors/sn4k3)

A high-performance .NET library that extends [Emgu.CV](https://www.emgu.com) (OpenCV wrapper) with zero-copy `Mat`
accessors, ROI utilities, async stream copy helpers, pluggable Mat compression, structured contour hierarchies, and
drawing helpers.

## Features

- **Zero-copy Mat access** - `Span<T>`, `ReadOnlySpan<T>`, `Memory<T>`, `ReadOnlyMemory<T>`, `Memory2D<T>`, and
  `ReadOnlyMemory2D<T>` accessors for fast pixel manipulation and async-friendly buffer usage
- **ROI utilities** - `Roi`, `SafeRoi`, `ConstrainRoi`, and `RoiFromCenter` for bounded region-of-interest cropping with
  configurable empty-ROI behavior
- **Image transforms** – Rotate with adjusted bounds, letterbox creation, crop-by-bounds, and shrink-to-fit resizing
  that reports whether the image changed, perspective crops, reusable destinations, and coordinate mapping
- **Stream copy helpers** - `CopyTo(Stream)` and `CopyToAsync(Stream, CancellationToken)` copy continuous matrices in
  one writing and non-continuous matrices row by row without including row padding
- **Alpha compositing** - `AlphaCompositeTo(background, offset, operation)` places a BGRA overlay onto a BGR/BGRA
  background with a Porter-Duff operator, clipped to the background bounds and written in place
- **Mask analysis** - `GetMaskComponents` returns connected regions (bounds, area, centroid) as managed values, and
  `CreateOklabDistanceMask` selects pixels perceptually close to a color
- **Image encoding** - `GetPngBytes` and `GetTiffBytes` (lossless for every common depth, selectable scheme and level)
- **Mat compression** - Pluggable compressors (PNG, Deflate, GZip, ZLib, Brotli, Zstd) with `CMat` for memory-efficient
  compressed image storage. `CMat.Compress` is thread-safe — concurrent calls are serialized automatically
- **Contour hierarchies** - Structured `EmguContour`, `EmguContours`, and `EmguContourFamily` wrappers for OpenCV
  contour trees using hierarchy links, plus area-filtered convex quadrilateral detection
- **Drawing helpers** - Polygon geometry, multi-line text rendering with alignment, SVG path generation, and predefined
  colors
- **Disposal infrastructure** - `DisposableObject`, `LeaveOpenDisposableObject`, and `GCSafeHandle` for thread-safe
  resource management

## Requirements

- [.NET 10](https://dotnet.microsoft.com/download) or later
- [Emgu.CV](https://www.nuget.org/packages/Emgu.CV) Latest

## OpenCV 5 migration notes

The projects use Emgu.CV 5.0.0.6584. See the [OpenCV 4 to 5 migration guide](https://github.com/opencv/opencv/wiki/OpenCV-4-to-5-migration).

- Affine and rotation helpers use `Mat` instead of the removed Emgu `Matrix<T>` type.
  OpenCV 5's revised warping improves accuracy and can improve performance without changing these helpers' APIs;
  pixel-exact output can differ from OpenCV 4.
- `DepthType.ByteCount` includes half and bfloat16, bool, uint32, uint64, and int64 storage.
  These sizes also support the library's raw memory and compression paths; image codecs still impose their own depth limits.
- `Emgu.CV.FontFace` overloads for `PutTextExtended`, `GetTextSizeExtended`, and `PutTextRotated` enable
  Unicode and custom fonts, with wrapping and text-block alignment. Existing overloads use `HersheyFonts`; legacy text appearance and measurements can change.
- Nearest-neighbor resize follows updated rounding rules, so old pixel baselines may need review.
  Vector-backed matrices can be genuinely 1D: use `Total` for element counts rather than assuming an Nx1 layout.
- The Windows native runtime packages currently produce NU1701 metadata warnings. The test project keeps these
  visible without treating them as errors; Windows x64 runtime behavior is covered by the test suite.

## Installation

### NuGet Package Manager

```
Install-Package EmguExtensions
```

### .NET CLI

```bash
dotnet add package EmguExtensions
```

## Quick Start

### Zero-Copy Mat Access

```csharp
using EmguExtensions;
using Emgu.CV;
using Emgu.CV.CvEnum;
using CommunityToolkit.HighPerformance;

// Create or load a Mat
using var mat = new Mat(100, 100, DepthType.Cv8U, 1);

// Zero-copy span access for fast pixel reads/writes
var span = mat.GetSpan<byte>();
span[0] = 255; // Set first pixel

// 2D span access
var span2D = mat.GetSpan2D<byte>();
span2D[10, 20] = 128; // Row 10, Col 20

// Memory access is useful when an API needs Memory<T> or ReadOnlyMemory<T>.
ReadOnlyMemory<byte> bytes = mat.GetReadOnlyMemoryOfBytes();
Memory<byte> writableBytes = mat.GetMemoryOfBytes();
writableBytes.Span[1] = 64;
```

`GetSpan<T>`, `GetMemory<T>`, and their read-only counterparts require a continuous `Mat`. Use the 2D accessors for ROI
views or other non-continuous matrices:

```csharp
using var source = new Mat(100, 100, DepthType.Cv8U, 1);
using var roi = new Mat(source, new Rectangle(10, 10, 40, 40)); // Usually non-continuous

var memory2D = roi.GetMemory2D<byte>();
memory2D.Span[0, 0] = 255;

ReadOnlyMemory2D<byte> readOnlyRoi = source.GetReadOnlyMemory2D<byte>(
    new Rectangle(10, 10, 40, 40));
byte first = readOnlyRoi.Span[0, 0];
```

### Unicode Text and Custom Fonts

Fonts are caller-owned and reusable. Font sizes are in pixels; `weight: 0` uses the font's current weight.
Use an embedded font (`"sans"`, `"italic"`, or `"uni"`) or pass a `.ttf`/`.otf` path to `FontFace`.
Available glyphs depend on the selected font. Text is shaped with HarfBuzz (OpenCV 5.1+), so complex scripts and
ligatures render correctly. The official Emgu.CV runtimes embed `"italic"` and `"uni"`; with a custom OpenCV build
that omits them, register the font files once with `FontFace.SetBuiltinFont("uni", path)` before creating fonts.

```csharp
using var font = new Emgu.CV.FontFace("uni");
using var image = EmguCvExtensions.InitMat(new Size(640, 480), channels: 3);
var white = new MCvScalar(255, 255, 255);

var size = EmguCvExtensions.GetTextSizeExtended(
    "Inspeção concluída\n你好", font, 28, wrapWidth: 300);
image.PutTextExtended("Inspeção concluída\n你好", new Point(20, 60), font, 28, white,
    wrapWidth: 300);
image.PutTextRotated("Etiqueta", new Point(320, 240), font, 28, white,
    lineAlignment: PutTextLineAlignment.Center, angle: 30);

// Returns the largest fitting size, or zero if the minimum size cannot fit.
int chosenSize = image.PutTextInRectangle("Inspeção concluída",
    new Rectangle(20, 300, 300, 100), font, white, maximumFontSize: 48);
```

The origin is the first line's baseline. Center and right alignment anchor the **whole text block** at `org.X`,
including wrapped lines. `wrapWidth: 0` disables wrapping. `PutTextInRectangle` fits and clips text inside a
rectangle wholly contained in the image, preserving pixels outside it. The existing Hershey-font APIs are retained.

### Perspective Cropping

Provide a strictly convex quadrilateral in **top-left, top-right, bottom-right, bottom-left** order.
The returned image is caller-owned; the source image is preserved.

```csharp
PointF[] corners = [new(40, 30), new(300, 50), new(320, 230), new(20, 210)];
using var rectified = image.CreatePerspectiveCrop(corners, new Size(280, 200));

// Without an explicit size, use the longer opposing edges, rounded up, plus one pixel.
using var inferred = image.CreatePerspectiveCrop(corners);
```

Corner coordinates describe pixel centers. Explicit output dimensions must each be at least two pixels.
Corners outside the source are permitted; optional `borderType` and `borderValue` control the outside samples.
Self-crossing, concave, repeated, collinear, and non-finite corners are rejected.

#### Ordering, Buffer Reuse, and Coordinate Mapping

```csharp
PointF[] unordered = [new(320, 230), new(40, 30), new(20, 210), new(300, 50)];
var ordered = EmguCvExtensions.OrderPerspectiveCorners(unordered);
var cropSize = EmguCvExtensions.GetPerspectiveCropSize(ordered);

using var destination = new Mat();
image.PerspectiveCrop(destination, ordered, cropSize);
// Reuses destination storage when dimensions and type match; overlapping buffers are handled safely.
image.PerspectiveCrop(destination, ordered, cropSize);

using var forward = EmguCvExtensions.CreatePerspectiveTransform(ordered, cropSize);
using var inverse = EmguCvExtensions.CreatePerspectiveTransform(ordered, cropSize, inverse: true);
var cropPoints = forward.MapPerspectivePoints(ordered);
var sourcePoints = inverse.MapPerspectivePoints(cropPoints);

// Crop a contour's minimum-area rotated rectangle directly.
var rotatedRectangle = new RotatedRect(new PointF(200, 150), new SizeF(120, 80), 25);
using var rectangleCrop = image.CreatePerspectiveCrop(rotatedRectangle);
```

Ordering uses clockwise image coordinates, starting at the smallest `X + Y`; ties use the smaller `Y`, then `X`.
This convention also determines the horizontal axis for rotated rectangles, including diamond shapes. Inputs are
preserved. Transform matrices are caller-owned; mapping rejects non-finite coordinates and points at infinity.

#### Quadrilateral Detection

```csharp
using var binary = EmguCvExtensions.InitMat(new Size(640, 480));
CvInvoke.Rectangle(binary, new Rectangle(50, 50, 300, 200), new MCvScalar(255), -1);
var candidates = binary.FindQuadrilaterals(minimumArea: 1000, approximationEpsilon: 0.02);
if (candidates.Length > 0)
{
    using var largest = image.CreatePerspectiveCrop(candidates[0]);
}
```

Pass a nonempty single-channel `Cv8U` binary image; threshold or otherwise segment your image first.
Candidates are convex four-corner contour approximations, ordered and sorted by decreasing polygon area.
`minimumArea` and `maximumArea` are inclusive filters; `approximationEpsilon` is a fraction of contour perimeter.
External contours are used by default; `retrievalMode` can include nested contours. The source is preserved,
and results are managed arrays requiring no disposal. This finds geometric candidates rather than identifying documents.

### Boolean Masks

```csharp
using var all = EmguCvExtensions.CreateBooleanMask(image.Size, value: true);
using var selectedBytes = binary.ToByteMask();
using var selected = selectedBytes.ToBooleanMask();
using var intersection = all.MaskAnd(selected);
using var union = all.MaskOr(selected);
using var exclusive = all.MaskXor(selected);
using var inverseMask = selected.MaskNot();

using var output = EmguCvExtensions.InitMat(image.Size, channels: image.NumberOfChannels, depthType: image.Depth);
image.CopyTo(output, intersection);

// For contours or codecs that require 8-bit masks:
using var byteMask = intersection.ToByteMask();
// Row-major bool input is also supported:
using var smallMask = EmguCvExtensions.CreateBooleanMask(new bool[] { true, false, false, true }, new Size(2, 2));
```

Logical operations accept nonempty single-channel `CvBool` or `Cv8U` masks, including non-continuous ROI views.
Every nonzero input means true. Binary operations require matching dimensions; results are caller-owned `CvBool`
images containing only `0` and `1`. `ToByteMask` produces `0` and `255`. Inputs are preserved, and logical negation
keeps canonical boolean values rather than flipping all storage bits.

### Safe ROI Cropping

```csharp
using var source = CvInvoke.Imread("image.png");

// SafeRoi clamps to mat bounds and updates the rectangle with the effective ROI.
var roiRect = new Rectangle(-10, -10, 200, 200);
using var roi = source.SafeRoi(ref roiRect);

// Padding can be applied while still keeping the ROI inside the source bounds.
var paddedRect = new Rectangle(40, 40, 120, 80);
using var paddedRoi = source.SafeRoi(ref paddedRect, padding: 16);

// ConstrainRoi only updates the rectangle and reports whether it changed.
var candidate = new Rectangle(0, 0, 500, 500);
bool changed = source.ConstrainRoi(ref candidate);

// Empty ROI behavior is explicit when a requested ROI becomes empty.
var emptyRect = Rectangle.Empty;
using var fullSource = source.SafeRoi(ref emptyRect, EmptyRoiBehavior.CaptureSource);

// Other options:
// EmptyRoiBehavior.Default        -> use OpenCV's default behavior for empty ROI
// EmptyRoiBehavior.CaptureSource  -> use the full source image
// EmptyRoiBehavior.ThrowException -> throw InvalidOperationException
```

### Image Sizing and Transforms

```csharp
// Create a fixed-size letterboxed image. targetWidth and targetHeight must be positive.
using var boxed = source.CreateLetterBox(640, 480, out float scale, out int padX, out int padY);

// Shrink only when the image exceeds the target bounds. Returns true when resized.
bool wasShrunk = source.ShrinkToFitPreserveAspect(1024, 768);

// Output overloads populate dst even when no resize/rotation is needed.
using var resized = new Mat();
bool resizedWasShrunk = source.ShrinkToFitPreserveAspect(resized, 1024, 768);

using var rotated = new Mat();
source.RotateAdjustBounds(rotated, angle: 30);
```

### Copy Mat Data to Streams

```csharp
using var output = File.Create("frame.raw");

// Synchronous copy.
mat.CopyTo(output);

// Async copy with cancellation support.
await using var asyncOutput = File.Create("frame-async.raw");
await mat.CopyToAsync(asyncOutput, cancellationToken);
```

For continuous matrices the whole data block is written in one operation. For non-continuous matrices, such as
partial-width ROI views, rows are written individually, so row padding is not included.

### Mat Compression with CMat

```csharp
using EmguExtensions;

// Compress a Mat for memory-efficient storage
var cmat = new CMat();
cmat.Compress(mat); // Auto-selects best storage (compressed or raw)

// Decompress back to Mat — caller owns and must dispose the returned Mat
using var restored = cmat.Decompress();

// Use a specific compressor and level
cmat.Compressor = MatCompressorBrotli.Instance;
cmat.CompressionLevel = CompressionLevel.SmallestSize;
cmat.Compress(mat);

// Compress(Mat) and Compress(MatRoi) are thread-safe — safe to call from multiple threads
Parallel.ForEach(frames, (frame, _, i) =>
{
    using var mat = frame.ToMat();
    compressedFrames[i].Compress(mat);
});

// Switch compressor and re-encode existing data in one step
cmat.ChangeCompressor(MatCompressorDeflate.Instance, reEncodeWithNewCompressor: true);

// Equality is hash-based (XxHash3) — O(1) for mismatches, fast for collections
bool same = cmat1 == cmat2;

// Direct compressor APIs also support async compression/decompression.
byte[] compressed = await MatCompressorBrotli.Instance.CompressAsync(mat, cancellationToken);
using var decompressed = mat.New();
await MatCompressorBrotli.Instance.DecompressAsync(compressed, decompressed, cancellationToken);
```

### Contour Hierarchy

```csharp
using EmguExtensions;

// Find contours with full hierarchy
using var contours = new EmguContours(binaryImage, RetrType.Tree);

// Iterate contour families (tree roots)
foreach (var family in contours.Families)
{
    Console.WriteLine($"Area: {family.Self.Area}, Children: {family.Count}");
    // Even depth = solid fill; odd depth = hole/cavity
}
```

### Drawing Helpers

```csharp
using EmguExtensions;

// Generate regular polygon vertices
var hexVertices = DrawingExtensions.GetPolygonVertices(6, new SizeF(50, 50), new PointF(100, 100));

// Multi-line text with alignment
mat.PutTextExtended("Line 1\nLine 2\nLine 3",
    new Point(50, 50),
    FontFace.HersheyComplex,
    1.0,
    EmguExtensions.WhiteColor,
    lineAlignment: PutTextLineAlignment.Center);
```

## Available Compressors

| Compressor | Class                  | Description                      |
|------------|------------------------|----------------------------------|
| None       | `MatCompressorNone`    | No compression (raw bytes)       |
| PNG        | `MatCompressorPng`     | PNG image encoding via OpenCV    |
| Deflate    | `MatCompressorDeflate` | Deflate stream compression       |
| GZip       | `MatCompressorGZip`    | GZip stream compression          |
| ZLib       | `MatCompressorZLib`    | ZLib stream compression          |
| Brotli     | `MatCompressorBrotli`  | Brotli stream compression        |
| Zstd       | `MatCompressorZstd`    | Zstandard compression (.NET 11+) |

All compressors are singletons accessed via `Instance` (e.g., `MatCompressorBrotli.Instance`). Public `CompressAsync`
and `DecompressAsync` methods call overrideable `CompressCoreAsync` and `DecompressCoreAsync` methods, so custom
compressors can provide native async behavior instead of only using the default background-thread wrapper.

`CompressionLevel` values are mapped to each compressor's native level range. For Brotli, `NoCompression`, `Fastest`,
`Optimal`, and `SmallestSize` map to qualities `0`, `1`, `4`, and `11` respectively.

`CMat` automatically falls back to raw (uncompressed) storage when:

- The source is smaller than `ThresholdToCompress` (default 512 bytes), or
- Compression produces a result larger than the original.

## Benchmarks

### Compressor benchmark

```
BenchmarkDotNet v0.15.8, Windows 11 (10.0.26200.8246/25H2/2025Update/HudsonValley2)
AMD Ryzen 9 7845HX with Radeon Graphics 3.00GHz, 1 CPU, 24 logical and 12 physical cores
.NET SDK 10.0.203
  [Host]     : .NET 10.0.7 (10.0.7, 10.0.726.21808), X64 RyuJIT x86-64-v4
  Job-MTJJIS : .NET 10.0.7 (10.0.7, 10.0.726.21808), X64 RyuJIT x86-64-v4

MaxIterationCount=16

| Method     | MatSize | CompressorName | Level         | Mean        | Error       | StdDev      | Gen0     | Gen1     | Gen2     | Allocated |
|----------- |-------- |--------------- |-------------- |------------:|------------:|------------:|---------:|---------:|---------:|----------:|
| Compress   | 1920    | Brotli         | NoCompression |    318.3 us |    14.55 us |    14.29 us |        - |        - |        - |    3200 B |
| Compress   | 1920    | Brotli         | Fastest       |    332.3 us |    54.77 us |    53.79 us |        - |        - |        - |    2816 B |
| Compress   | 1920    | None           | NoCompression |    337.8 us |    11.88 us |    11.66 us | 194.3359 | 194.3359 | 194.3359 | 3686483 B |
| Compress   | 1920    | Deflate        | Fastest       |    636.3 us |    55.72 us |    54.73 us |   1.9531 |        - |        - |   45744 B |
| Compress   | 1920    | GZip           | Fastest       |    638.2 us |    28.39 us |    27.88 us |   2.4414 |        - |        - |   45792 B |
| Compress   | 1920    | ZLib           | Fastest       |    640.3 us |    49.51 us |    48.62 us |   2.4414 |        - |        - |   45784 B |
| Compress   | 1920    | GZip           | NoCompression |    769.6 us |    13.23 us |    12.38 us | 276.3672 | 276.3672 | 276.3672 | 3688281 B |
| Compress   | 1920    | ZLib           | NoCompression |    770.1 us |    27.85 us |    27.35 us | 284.1797 | 284.1797 | 284.1797 | 3688256 B |
| Compress   | 1920    | Deflate        | NoCompression |    862.5 us |    14.73 us |    13.78 us | 280.2734 | 280.2734 | 280.2734 | 3688228 B |
| Compress   | 1920    | Deflate        | Optimal       |  1,303.5 us |    42.47 us |    41.71 us |   0.9766 |        - |        - |   23512 B |
| Compress   | 1920    | GZip           | Optimal       |  1,305.6 us |    93.40 us |    91.74 us |        - |        - |        - |   23568 B |
| Compress   | 1920    | ZLib           | Optimal       |  1,353.0 us |    67.70 us |    66.49 us |        - |        - |        - |   23552 B |
| Compress   | 1920    | Brotli         | Optimal       |  4,997.6 us |   181.04 us |   177.80 us |        - |        - |        - |    1313 B |
| Compress   | 1920    | GZip           | SmallestSize  |  8,240.1 us |   397.06 us |   389.97 us |        - |        - |        - |   15552 B |
| Compress   | 1920    | ZLib           | SmallestSize  |  8,380.9 us |   372.70 us |   366.04 us |        - |        - |        - |   15536 B |
| Compress   | 1920    | Deflate        | SmallestSize  |  8,492.4 us |   427.02 us |   419.39 us |        - |        - |        - |   15504 B |
| Compress   | 1920    | PNG            | Fastest       | 15,511.5 us | 1,415.48 us | 1,390.19 us |        - |        - |        - |   25736 B |
| Compress   | 1920    | PNG            | NoCompression | 15,571.9 us |   654.92 us |   643.22 us | 375.0000 | 375.0000 | 375.0000 | 3694733 B |
| Compress   | 1920    | PNG            | Optimal       | 16,101.5 us |   679.06 us |   666.93 us |        - |        - |        - |   25496 B |
| Compress   | 1920    | PNG            | SmallestSize  | 35,974.1 us | 2,452.16 us | 2,408.35 us |        - |        - |        - |    9280 B |
| Compress   | 1920    | Brotli         | SmallestSize  | 66,692.0 us | 9,677.48 us | 9,504.58 us |        - |        - |        - |     692 B |
|            |         |                |               |             |             |             |          |          |          |           |
| Decompress | 1920    | Deflate        | SmallestSize  |    113.4 us |     1.35 us |     1.19 us |        - |        - |        - |     280 B |
| Decompress | 1920    | Deflate        | Fastest       |    119.7 us |     2.57 us |     2.52 us |        - |        - |        - |     280 B |
| Decompress | 1920    | Deflate        | Optimal       |    123.6 us |     3.56 us |     3.50 us |        - |        - |        - |     280 B |
| Decompress | 1920    | Deflate        | NoCompression |    145.0 us |     2.05 us |     1.82 us |        - |        - |        - |     280 B |
| Decompress | 1920    | None           | NoCompression |    147.1 us |     1.93 us |     1.71 us |        - |        - |        - |         - |
| Decompress | 1920    | ZLib           | Fastest       |    157.1 us |     1.58 us |     1.32 us |        - |        - |        - |     312 B |
| Decompress | 1920    | GZip           | Fastest       |    161.6 us |     2.30 us |     1.92 us |        - |        - |        - |     312 B |
| Decompress | 1920    | GZip           | Optimal       |    162.2 us |     2.07 us |     1.93 us |        - |        - |        - |     312 B |
| Decompress | 1920    | GZip           | SmallestSize  |    168.2 us |     8.01 us |     7.86 us |        - |        - |        - |     312 B |
| Decompress | 1920    | ZLib           | Optimal       |    177.4 us |     1.59 us |     1.41 us |        - |        - |        - |     312 B |
| Decompress | 1920    | GZip           | NoCompression |    182.6 us |     1.61 us |     1.50 us |        - |        - |        - |     312 B |
| Decompress | 1920    | ZLib           | SmallestSize  |    197.6 us |     8.23 us |     8.08 us |        - |        - |        - |     312 B |
| Decompress | 1920    | ZLib           | NoCompression |    225.5 us |    13.96 us |    13.71 us |        - |        - |        - |     312 B |
| Decompress | 1920    | Brotli         | NoCompression |  1,560.4 us |    24.39 us |    21.62 us |        - |        - |        - |         - |
| Decompress | 1920    | Brotli         | Fastest       |  2,107.5 us |   267.52 us |   262.74 us |        - |        - |        - |         - |
| Decompress | 1920    | Brotli         | SmallestSize  |  2,354.4 us |    49.89 us |    49.00 us |        - |        - |        - |         - |
| Decompress | 1920    | Brotli         | Optimal       |  3,384.4 us |   254.81 us |   250.26 us |        - |        - |        - |         - |
| Decompress | 1920    | PNG            | Fastest       |  6,079.8 us |    39.73 us |    35.22 us |        - |        - |        - |      88 B |
| Decompress | 1920    | PNG            | Optimal       |  6,120.8 us |    43.09 us |    38.19 us |        - |        - |        - |      88 B |
| Decompress | 1920    | PNG            | NoCompression |  6,830.0 us |    47.14 us |    44.09 us |        - |        - |        - |      88 B |
| Decompress | 1920    | PNG            | SmallestSize  |  8,797.8 us |    39.41 us |    34.94 us |        - |        - |        - |      88 B |
```

### Mat.ToArray() benchmark

```
BenchmarkDotNet v0.15.8, Windows 11 (10.0.26200.8246/25H2/2025Update/HudsonValley2)
AMD Ryzen 9 7845HX with Radeon Graphics 3.00GHz, 1 CPU, 24 logical and 12 physical cores
.NET SDK 10.0.202
  [Host]     : .NET 10.0.6 (10.0.6, 10.0.626.17701), X64 RyuJIT x86-64-v4
  DefaultJob : .NET 10.0.6 (10.0.6, 10.0.626.17701), X64 RyuJIT x86-64-v4


| Method                | MatSize | Mean       | Error    | StdDev   | Ratio | RatioSD | Gen0     | Gen1     | Gen2     | Allocated | Alloc Ratio |
|---------------------- |-------- |-----------:|---------:|---------:|------:|--------:|---------:|---------:|---------:|----------:|------------:|
| GetRawData            | 1920    |   617.7 us |  6.08 us |  5.08 us |  0.99 |    0.04 | 999.0234 | 999.0234 | 999.0234 |   3.52 MB |        1.00 |
| GetSpan.ToArray       | 1920    |   623.8 us | 10.47 us |  9.28 us |  1.00 |    0.05 | 999.0234 | 999.0234 | 999.0234 |   3.52 MB |        1.00 |
| 'ToArray (extension)' | 1920    |   626.5 us | 11.84 us | 28.15 us |  1.00 |    0.06 | 999.0234 | 999.0234 | 999.0234 |   3.52 MB |        1.00 |
|                       |         |            |          |          |       |         |          |          |          |           |             |
| 'ToArray (extension)' | 3840    | 1,444.4 us | 28.81 us | 32.02 us |  1.00 |    0.03 | 500.0000 | 500.0000 | 500.0000 |  14.06 MB |        1.00 |
| GetSpan.ToArray       | 3840    | 1,551.3 us | 16.07 us | 15.04 us |  1.07 |    0.02 | 500.0000 | 500.0000 | 500.0000 |  14.06 MB |        1.00 |
| GetRawData            | 3840    | 1,582.3 us | 21.02 us | 19.66 us |  1.10 |    0.03 | 500.0000 | 500.0000 | 500.0000 |  14.06 MB |        1.00 |
```

## Building from Source

```bash
# Restore and build
dotnet restore
dotnet build

# Run tests
dotnet test EmguExtensions.Tests

# Pack for NuGet (Release only)
dotnet pack --configuration Release --output .
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
