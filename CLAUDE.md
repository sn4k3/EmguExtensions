# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build, Test & Pack

```bash
dotnet restore
dotnet build
dotnet test EmguExtensions.Tests
dotnet test EmguExtensions.Tests --filter "FullyQualifiedName~UnitTestCMat"       # filter by class
dotnet test EmguExtensions.Tests --filter "FullyQualifiedName~Compress_Empty"     # filter by method substring
dotnet pack EmguExtensions --configuration Release --output .
```

Packing is disabled under `Debug` configuration (`IsPackable=false`). Build artifacts go to `artifacts/`.

## Code Conventions

- **C# latest (.NET 10)** with `<Nullable>enable</Nullable>` and file-scoped namespaces (`namespace EmguExtensions;`)
- **XML doc comments** (`///`) required on all public members
- **Private fields**: `_camelCase`
- **`#region`** blocks for organization within large files
- **`unsafe` blocks** are allowed (`AllowUnsafeBlocks=true`)
- Extensions use C# 14's explicit extension block syntax: `extension(Mat mat) { ... }` inside a `static class`
- Factory methods are prefixed `Create`; measurement/query methods are prefixed `Get`

## Architecture

Single-project library (`EmguExtensions/`) providing extension methods and helpers for Emgu.CV (OpenCV wrapper). Organized into five main subsystems:

### Extensions (`Extensions/`)

The primary partial class `EmguCvExtensions` is split across several files:

- **`EmguCvExtensions.cs`** — `extension(Mat mat)` block with span accessors (`GetSpan<T>`, `GetSpan2D<T>`, `GetReadOnlySpan2D<T>`), ROI helpers (`Roi`, `SafeRoi`, `RoiFromCenter`), copy utilities (`CopyTo`, `CopyToCenter`), unsafe raw-pointer accessors (`BytePointer`, `GetUnmanagedMemoryStream`), image transforms (`CreateLetterBox`), and drawing helpers (`GetSvgPath`, `DrawLineAccurate`, `PutTextExtended`).
- **`EmguCvExtensions.Constants.cs`** — Predefined `MCvScalar` colors, `AnchorCenter`, and the shared read-only `Kernel3X3Rectangle` (never dispose or modify it; `Clone()` it for a private copy).
- **`EmguCvExtensions.Static.cs`** — Static helpers: `GetTextSizeExtended()` (multiline text measurement with alignment), `PutTextLineAlignmentTrim()`, `CorrectThickness()`, `CreateDynamicKernel()`, `InitMat()` factory overloads, plus private helpers shared by the extension block (`ResolveElementRange`, vertical `ScanLines` and SVG number formatting).
- **`StreamExtensions.cs` / `DotNextExtensions.cs`** — `MemoryStream.ToArrayPerf()` and `SparseBufferWriter<T>.ToArray()` used by the compressors.
- **`DrawingExtensions.cs`** — Color scaling (`FactorColor`), regular-polygon geometry (`CalculatePolygonSideLengthFromRadius`, `CalculatePolygonRadiusFromSideLength`), and vertex generation (`GetPolygonVertices`, `GetAlignedPolygonVertices`).
- **`PointExtensions.cs`** — Euclidean distance (`FindLength`), `Point`/`PointF` rotation around a pivot.
- **`ArrayExtensions.cs`** — `ToArrayPerf()`: uninitialized-memory array copy for performance-sensitive paths.
- **`CompressionExtensions.cs`** — `GetCompressionLevel(int)`: maps 0–3 to `CompressionLevel` enum values.

**ROI safety pattern:** Always prefer `SafeRoi` over `Roi` when coordinates may be out of bounds — it clamps to matrix dimensions and returns an empty `Mat` when the clamped result is zero-sized.

### Mat Compression (`MatCompressor/`)

Abstract base `MatCompressor` implements the template-method pattern:
- `Compress(Mat, CompressionLevel)` guards for empty `Mat` then calls `CompressCore` (abstract).
- `Decompress(byte[], Mat)` guards for empty bytes, requires an already-allocated `dst` (throws `ArgumentException` otherwise, unless the compressor sets `AllocatesDestination`, e.g. PNG) then calls `DecompressCore` (abstract). Corrupt/mismatching data throws `InvalidDataException`.
- All concrete implementations are **singletons** accessed via `Instance`.
- When adding a new compressor, implement `CompressCore` / `DecompressCore` — never override the non-virtual `Compress(Mat, CompressionLevel)` or `Decompress(byte[], Mat)` directly.

Available compressors (registered in `MatCompressor.AvailableCompressors`): `None`, `PNG`, `Deflate`, `GZip`, `ZLib`, `Brotli`, `Zstd` (.NET 11+ only, guarded by `#if NET11_0_OR_GREATER`).

**`CMat`** is a compressed-Mat class (`IEquatable<CMat>`):
- Stores `CompressedBytes` and tracks `Compressor`/`CompressionLevel` separately from `Decompressor` (the algorithm used to create the stored bytes — may differ after `ChangeCompressor`).
- `Compress(Mat)` auto-selects raw storage when compression is larger than (or an empty result for) the source, the source is below `ThresholdToCompress` (default 512 bytes), the compressor is `None`, or the codec fails (`ArgumentException`s such as an invalid level still propagate).
- `Decompress()` reconstructs the full-size `Mat`, expanding into the original dimensions when `Roi` is set.
- `RawDecompress()` returns only the ROI-sized slice without expanding.
- Equality is hash-based (`XxHash3` over `CompressedBytes`, cached) — O(1) for non-matching lengths. `CompressedBytes` is the internal array: treat it as read-only. `SetCompressedBytes` takes ownership of the array and clears `Roi`; `Clone`/`CopyTo` deep-copy the bytes.
- `ChangeCompressor(compressor, reEncodeWithNewCompressor: true)` re-encodes existing bytes with the new compressor in one atomic step.
- **Thread safety**: Uses `ReaderWriterLockSlim` for multiple-reader/single-writer concurrency. All state lives in private fields: every public property read takes the read lock, every mutation (`Compress`, `SetEmptyCompressedBytes`, `SetCompressedBytes`, `ChangeCompressor`, the `Compressor`/`CompressionLevel`/`ThresholdToCompress` setters) takes the write lock, and multi-value members (ratios, `Equals`, `GetHashCode`, `ToString`, `CopyTo`) work on a single `GetState()` snapshot. Internal unlocked methods (`CompressInternal`, `RawDecompressInternal`, `CreateMatInternal`) use the fields directly: never call a public property from inside a locked region (`ReaderWriterLockSlim` does not allow recursion).
- `Decompress()` returns a newly allocated caller-owned `Mat` — always dispose it.

### Contours (`Contours/`)

Structured wrappers for OpenCV contour hierarchies. Best used with `RetrType.Tree`.

- **`EmguContour`** — Wraps a single `VectorOfPoint`. Lazy-computes `Bounds`, `BoundsBestFit`, `MinEnclosingCircle`, area, perimeter, and convexity. Hierarchy index constants (`HierarchyNextSameLevel`, `HierarchyParent`, etc.) are fields. Extends `DisposableObject`.
- **`EmguContours`** — Wraps `VectorOfVectorOfPoint` + hierarchy matrix. Implements `IReadOnlyList<EmguContour>`. Exposes `Families` (tree roots) and `ExternalContoursCount`. Extends `LeaveOpenDisposableObject`.
- **`EmguContourFamily`** — Tree node with `Self` (EmguContour), `Depth`, `Parent`, and `Children`. Even depth = solid fill; odd depth = hole/cavity.

### Strides (`Strides/`)

Lightweight pixel-scan result types returned by scan/find extension methods:

- **`GreyLine`** — `record struct` representing a straight horizontal run of pixels sharing the same grey value. Fields: `StartX`, `StartY`, `EndX`, `EndY` and `Grey` (it also describes vertical runs).
- **`GreyStride`** — `readonly record struct` representing a contiguous run anywhere in the image. Constructor parameters: `Index` (flat offset), `Location` (`Point`), `Stride` (length), `Grey`.

### Handlers (`Handlers/`)

Reusable disposal infrastructure:

- **`DisposableObject`** — Abstract base for the full Dispose pattern. Subclasses override `protected abstract DisposeManaged()` and optionally `protected virtual DisposeUnmanaged()`.
- **`LeaveOpenDisposableObject`** — Extends `DisposableObject` with a `LeaveOpen` init property; when `true`, owned resources survive disposal (caller retains ownership).
- **`GCSafeHandle`** — `SafeHandle` wrapper for `GCHandle`. Pins managed memory for P/Invoke without manual cleanup risk.

### Other

- **`MatRoi`** — ROI crop with optional per-edge padding. Extends `LeaveOpenDisposableObject`. Uses `SafeRoi()` internally; exposes `SourceMat`, `RoiMat`, `Roi`, and `IsSourceSameSizeOfRoi`.
- **`PutTextLineAlignment`** — Enum: `Default | Left | Center | Right` (`Default` only trims the end of each line). Controls trimming and horizontal layout in `PutTextExtended`.
- **`StaticObjects`** — Internal; holds `LineBreakCharacters` (`["\r\n", "\r", "\n"]`) for multiline text splitting.

### Avalonia (`EmguExtensions.Avalonia/`)

Separate package with two static classes (namespace `EmguExtensions.Avalonia`):

- **`EmguCvAvaloniaExtensions`** — `Mat.ToBitmap(...)`/`ToBitmapAsync(...)` overloads (default, `Type srcType`, `scale/shift`). All of them go through `PrepareSource` (validation, optional 8-bit scaling) and `CreateBitmap`, which converts directly into the locked framebuffer through a `Mat` view (`ILockedFramebuffer.ToMat()`), so there is no intermediate Mat. Output is always `Bgra8888`/`Unpremul`.
- **`AvaloniaBitmapExtensions`** — span/`Span2D`/`Mat` views over `ILockedFramebuffer` (stride aware, `System.Drawing` and `PixelPoint`/`PixelRect` overloads) and `WriteableBitmap` helpers (`GetBitmapInfo` without address, `WithBitmapInfo`, `ToMat`). Anything that exposes pixel memory is only valid while the framebuffer is locked; dispose a `Mat` view before the framebuffer.
- Tests live in `EmguExtensions.Tests/UnitTestAvalonia.cs` and run on the Avalonia headless platform with **Skia** (`UseHeadlessDrawing = false`): the headless drawing stub does not keep pixels between `Lock()` calls.
- The namespace `EmguExtensions.Avalonia` shadows the real `Avalonia` namespace inside `EmguExtensions.*` code: use `using` directives/aliases there instead of qualified `Avalonia.X` names.

## Project Metadata

Central metadata (version, authors, NuGet config, artifact paths) lives in `Directory.Build.props`. The assembly is strong-name signed; `EmguExtensions.snk` must be present to build.

## Custom Claude Commands (`.claude/commands/`)

Project-specific slash commands available in this session:

| Command | Purpose |
|---------|---------|
| `/review` | 7-section code review (bugs, leaks, nullability, validation, performance, unsafe, API design) |
| `/document` | Add XML doc comments to all public members |
| `/guards` | Add parameter validation guards to public methods |
| `/optimize` | Allocation and performance optimization pass |
| `/unsafe-audit` | Audit all `unsafe` blocks for correctness |
| `/test` | Generate xUnit tests with edge cases |
| `/dispose` | Audit IDisposable correctness and resource leaks |
| `/nullsafe` | Audit nullable reference type correctness |
