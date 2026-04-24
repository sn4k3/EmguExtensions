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

The primary partial class `EmguExtensions` is split across several files:

- **`EmguExtensions.cs`** — `extension(Mat mat)` block with span accessors (`GetSpan<T>`, `GetSpan2D<T>`, `GetReadOnlySpan2D<T>`), ROI helpers (`Roi`, `SafeRoi`, `RoiFromCenter`), copy utilities (`CopyTo`, `CopyToCenter`), unsafe raw-pointer accessors (`BytePointer`, `GetUnmanagedMemoryStream`), image transforms (`CreateLetterBox`), and drawing helpers (`GetSvgPath`, `DrawLineAccurate`, `PutTextExtended`).
- **`EmguExtensions.Constants.cs`** — Predefined `MCvScalar` colors, `AnchorCenter`, and the shared `RecyclableMemoryStreamManager` singleton for stream pooling across all compressors.
- **`EmguExtensions.Static.cs`** — Static helpers: `GetTextSizeExtended()` (multiline text measurement with alignment), `PutTextLineAlignmentTrim()`, `CorrectThickness()`, `CreateDynamicKernel()`, and `InitMat()` factory overloads.
- **`DrawingExtensions.cs`** — Color scaling (`FactorColor`), regular-polygon geometry (`CalculatePolygonSideLengthFromRadius`, `CalculatePolygonRadiusFromSideLength`), and vertex generation (`GetPolygonVertices`, `GetAlignedPolygonVertices`).
- **`PointExtensions.cs`** — Euclidean distance (`FindLength`), `Point`/`PointF` rotation around a pivot.
- **`ArrayExtensions.cs`** — `ToArrayPerf()`: uninitialized-memory array copy for performance-sensitive paths.
- **`CompressionExtensions.cs`** — `GetCompressionLevel(int)`: maps 0–3 to `CompressionLevel` enum values.

**ROI safety pattern:** Always prefer `SafeRoi` over `Roi` when coordinates may be out of bounds — it clamps to matrix dimensions and returns an empty `Mat` when the clamped result is zero-sized.

### Mat Compression (`MatCompressor/`)

Abstract base `MatCompressor` implements the template-method pattern:
- `Compress(Mat, CompressionLevel)` guards for empty `Mat` then calls `CompressCore` (abstract).
- `Decompress(byte[], Mat)` guards for empty bytes then calls `DecompressCore` (abstract).
- All concrete implementations are **singletons** accessed via `Instance`.
- When adding a new compressor, implement `CompressCore` / `DecompressCore` — never override the non-virtual `Compress(Mat, CompressionLevel)` or `Decompress(byte[], Mat)` directly.

Available compressors (registered in `MatCompressor.AvailableCompressors`): `None`, `PNG`, `Deflate`, `GZip`, `ZLib`, `Brotli`, `Zstd` (.NET 11+ only, guarded by `#if NET11_0_OR_GREATER`).

**`CMat`** is a compressed-Mat class (`IEquatable<CMat>`):
- Stores `CompressedBytes` and tracks `Compressor`/`CompressionLevel` separately from `Decompressor` (the algorithm used to create the stored bytes — may differ after `ChangeCompressor`).
- `Compress(Mat)` auto-selects raw storage when compression is larger than the source or the source is below `ThresholdToCompress` (default 512 bytes).
- `Decompress()` reconstructs the full-size `Mat`, expanding into the original dimensions when `Roi` is set.
- `RawDecompress()` returns only the ROI-sized slice without expanding.
- Equality is hash-based (`XxHash3` over `CompressedBytes`) — O(1) for non-matching lengths.
- `ChangeCompressor(compressor, reEncodeWithNewCompressor: true)` re-encodes existing bytes with the new compressor in one atomic step.
- **Thread safety**: Uses `ReaderWriterLockSlim` for multiple-reader/single-writer concurrency. Write operations (`Compress`, `SetEmptyCompressedBytes`, `SetCompressedBytes`, `ChangeCompressor`) acquire a write lock; read operations (`Decompress`, `RawDecompress`, `CopyTo`, `Hash`) acquire a read lock. Internal unlocked methods (`CompressInternal`, `RawDecompressInternal`) avoid recursive locking in composite operations.
- `Decompress()` returns a newly allocated caller-owned `Mat` — always dispose it.

### Contours (`Contours/`)

Structured wrappers for OpenCV contour hierarchies. Best used with `RetrType.Tree`.

- **`EmguContour`** — Wraps a single `VectorOfPoint`. Lazy-computes `Bounds`, `BoundsBestFit`, `MinEnclosingCircle`, area, perimeter, and convexity. Hierarchy index constants (`HierarchyNextSameLevel`, `HierarchyParent`, etc.) are fields. Extends `DisposableObject`.
- **`EmguContours`** — Wraps `VectorOfVectorOfPoint` + hierarchy matrix. Implements `IReadOnlyList<EmguContour>`. Exposes `Families` (tree roots) and `ExternalContoursCount`. Extends `LeaveOpenDisposableObject`.
- **`EmguContourFamily`** — Tree node with `Self` (EmguContour), `Depth`, `Parent`, and `Children`. Even depth = solid fill; odd depth = hole/cavity.

### Strides (`Strides/`)

Lightweight pixel-scan result types returned by scan/find extension methods:

- **`GreyLine`** — `record struct` representing a straight horizontal run of pixels sharing the same grey value. Fields: `StartX`, `Y`, `EndX`, `Grey`, and `Length`.
- **`GreyStride`** — `readonly record struct` representing a contiguous run anywhere in the image. Constructor parameters: `Index` (flat offset), `Location` (`Point`), `Stride` (length), `Grey`.

### Handlers (`Handlers/`)

Reusable disposal infrastructure:

- **`DisposableObject`** — Abstract base for the full Dispose pattern. Subclasses override `protected abstract DisposeManaged()` and optionally `protected virtual DisposeUnmanaged()`.
- **`LeaveOpenDisposableObject`** — Extends `DisposableObject` with a `LeaveOpen` init property; when `true`, owned resources survive disposal (caller retains ownership).
- **`GCSafeHandle`** — `SafeHandle` wrapper for `GCHandle`. Pins managed memory for P/Invoke without manual cleanup risk.

### Other

- **`MatRoi`** — ROI crop with optional per-edge padding. Extends `LeaveOpenDisposableObject`. Uses `SafeRoi()` internally; exposes `SourceMat`, `RoiMat`, `Roi`, and `IsSourceSameSizeOfRoi`.
- **`PutTextLineAlignment`** — Enum: `None | Left | Center | Right`. Controls trimming and horizontal layout in `PutTextExtended`.
- **`StaticObjects`** — Internal; holds `LineBreakCharacters` (`["\r\n", "\r", "\n"]`) for multiline text splitting.

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
