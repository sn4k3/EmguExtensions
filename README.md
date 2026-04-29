# [![Logo](https://raw.githubusercontent.com/sn4k3/EmguExtensions/main/media/EmguExtensions-32.png)](#) EmguExtensions

[![License](https://img.shields.io/github/license/sn4k3/EmguExtensions?style=for-the-badge)](https://github.com/sn4k3/EmguExtensions/blob/master/LICENSE)
[![GitHub repo size](https://img.shields.io/github/repo-size/sn4k3/EmguExtensions?style=for-the-badge)](#)
[![Code size](https://img.shields.io/github/languages/code-size/sn4k3/EmguExtensions?style=for-the-badge)](#)
[![Nuget](https://img.shields.io/nuget/v/EmguExtensions?style=for-the-badge)](https://www.nuget.org/packages/EmguExtensions)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/sn4k3?color=red&style=for-the-badge)](https://github.com/sponsors/sn4k3)

A high-performance .NET library that extends [Emgu.CV](https://www.emgu.com) (OpenCV wrapper) with span-based `Mat` accessors, ROI utilities, pluggable Mat compression, structured contour hierarchies, and drawing helpers.

## Features

- **Span-based Mat access** - Zero-copy `GetSpan<T>`, `GetSpan2D<T>`, and `GetReadOnlySpan2D<T>` accessors for fast pixel manipulation
- **ROI utilities** - `Roi`, `SafeRoi` (clamped), and `RoiFromCenter` for safe region-of-interest cropping
- **Image transforms** - Rotate with adjusted bounds, letterbox creation, crop-by-bounds, and shrink-to-fit resizing that reports whether the image changed
- **Mat compression** - Pluggable compressors (PNG, Deflate, GZip, ZLib, Brotli, Zstd) with `CMat` for memory-efficient compressed image storage. `CMat.Compress` is thread-safe — concurrent calls are serialized automatically
- **Contour hierarchies** - Structured `EmguContour`, `EmguContours`, and `EmguContourFamily` wrappers for OpenCV contour trees using hierarchy links
- **Drawing helpers** - Polygon geometry, multi-line text rendering with alignment, SVG path generation, and predefined colors
- **Disposal infrastructure** - `DisposableObject`, `LeaveOpenDisposableObject`, and `GCSafeHandle` for thread-safe resource management

## Requirements

- [.NET 10](https://dotnet.microsoft.com/download) or later
- [Emgu.CV](https://www.nuget.org/packages/Emgu.CV) Latest

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

### Span-Based Mat Access

```csharp
using EmguExtensions;
using Emgu.CV;
using Emgu.CV.CvEnum;

// Create or load a Mat
using var mat = new Mat(100, 100, DepthType.Cv8U, 1);

// Zero-copy span access for fast pixel reads/writes
var span = mat.GetSpan<byte>();
span[0] = 255; // Set first pixel

// 2D span access
var span2D = mat.GetSpan2D<byte>();
span2D[10, 20] = 128; // Row 10, Col 20
```

### Safe ROI Cropping

```csharp
using var source = CvInvoke.Imread("image.png");

// SafeRoi clamps to mat bounds - never throws on out-of-range coordinates
using var roi = source.SafeRoi(new Rectangle(-10, -10, 200, 200));
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

| Compressor | Class | Description |
|---|---|---|
| None | `MatCompressorNone` | No compression (raw bytes) |
| PNG | `MatCompressorPng` | PNG image encoding via OpenCV |
| Deflate | `MatCompressorDeflate` | Deflate stream compression |
| GZip | `MatCompressorGZip` | GZip stream compression |
| ZLib | `MatCompressorZLib` | ZLib stream compression |
| Brotli | `MatCompressorBrotli` | Brotli stream compression |
| Zstd | `MatCompressorZstd` | Zstandard compression (.NET 11+) |

All compressors are singletons accessed via `Instance` (e.g., `MatCompressorBrotli.Instance`).  

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

## Project Structure

```
EmguExtensions/
  Extensions/
    EmguExtensions.cs           # Mat span accessors, ROI, copy, transforms, drawing, shrink-to-fit
    EmguExtensions.Constants.cs # Predefined colors, anchor, shared stream manager
    EmguExtensions.Static.cs    # Text measurement, kernel creation, Mat factories
    DrawingExtensions.cs        # Color scaling, polygon geometry and vertices
    PointExtensions.cs          # Euclidean distance, rotation around a pivot
    ArrayExtensions.cs          # High-performance uninitialized-memory array copy
    StreamExtensions.cs         # MemoryStream.ToArrayPerf — copy buffer without zero-init
    CompressionExtensions.cs    # CompressionLevel enum mapping (0–3 → enum)
    StaticObjects.cs            # Internal: line-break character constants
  MatCompressor/
    MatCompressor.cs            # Abstract base — template method, async overloads
    MatCompressor.None.cs       # Raw (uncompressed) passthrough
    MatCompressor.Png.cs        # PNG via OpenCV Imdecode/Imencode
    MatCompressor.Deflate.cs    # Deflate stream
    MatCompressor.GZip.cs       # GZip stream
    MatCompressor.ZLib.cs       # ZLib stream
    MatCompressor.Brotli.cs     # Brotli stream
    MatCompressor.Zstd.cs       # Zstandard stream (.NET 11+)
    CMat.cs                     # Compressed Mat — thread-safe Compress, XxHash3 equality
  Contours/
    EmguContour.cs              # Single contour wrapper (lazy bounds, area, perimeter, disposal guards)
    EmguContours.cs             # Contour collection with hierarchy-link tree and grouping helpers
    EmguContourFamily.cs        # Tree node: Self, Depth, Parent, Count/children traversal
  Handlers/
    DisposableObject.cs         # Abstract full Dispose pattern base with atomic dispose guard
    LeaveOpenDisposableObject.cs # Dispose with leave-open semantics
    GCSafeHandle.cs             # SafeHandle wrapper for GCHandle pinning
  Strides/
    GreyLine.cs                 # Straight run of pixels with the same grey value (line scan)
    GreyStride.cs               # Contiguous pixel run with index, location, and grey value
  MatRoi.cs                     # ROI crop with optional per-edge padding
  Enums/
    PutTextLineAlignment.cs     # None | Left | Center | Right for PutTextExtended
EmguExtensions.Tests/           # xUnit test suite
  MatFactory.cs                 # Shared Mat helpers for tests
  UnitTestCMat.cs               # CMat compression/decompression tests
  UnitTestCMatThreadSafety.cs   # CMat concurrent Compress thread-safety tests
  UnitTestEmguContours.cs       # Contour hierarchy tests
  UnitTestEmguExtensions.cs     # Extension method tests
  UnitTestFindMethods.cs        # Pixel-find method tests
  UnitTestMatRoi.cs             # MatRoi crop and padding tests
  UnitTestScanMethods.cs        # Scan/stride method tests
EmguExtensions.Benchmarks/      # BenchmarkDotNet performance benchmarks
  MatCompressorBenchmarks.cs    # Compress/Decompress measurements across compressors and levels
  MatToArrayBenchmarks.cs       # Mat-to-array conversion micro-benchmarks
  RedundantCompressorLevelFilter.cs # Filter to skip redundant compressor/level pairs
  Program.cs                    # Benchmark runner entry point
```

## Building from Source

```bash
# Restore and build
dotnet restore
dotnet build

# Run tests
dotnet test EmguExtensions.Tests

# Pack for NuGet (Release only)
dotnet pack EmguExtensions --configuration Release --output .
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
