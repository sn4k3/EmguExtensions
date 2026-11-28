# v0.4.2 (28/11/2026)

- Add `MatCompressorTiff` (registered in `MatCompressor.AvailableCompressors`): lossless TIFF codec via OpenCV with
  Adobe Deflate (levels 1-9, 0 stores uncompressed, `Optimal` = 3). Best for continuous-tone data (photos, depth
  maps, float fields), where its predictor compresses far better than the stream compressors. Unlike PNG it compresses signed and 32/64-bit integer and `Cv32F`
  matrices. Depths it cannot store losslessly (`Cv16F`, `Cv16BF`, `CvBool`, FP8) and channel counts other than 1, 3 or 4 throw `NotSupportedException`, so `CMat` stores them raw. OpenCV writes `Cv64F` uncompressed.
- Add `MatCompressorDelta`: wraps a byte compressor with a lossless delta filter (each element minus the previous one
  of the same channel in the row, then byte planes for multi-byte depths; vectorized with SSSE3/ARM64 shuffles and
  undone in place on decompression, adding only about 1-6 ms over the inner compressor at 4K). Supports every depth, channel count, ROI
  view and non-continuous matrix. On continuous-tone data `MatCompressorDelta.Zstd` compresses 1.7-1.9× versus about
  1.05× for plain Zstd, matching or beating `MatCompressorTiff` while being 2-5× faster. Ready-made `Deflate`,
  `Brotli` and `Zstd` (.NET 11+) variants are registered in `MatCompressor.AvailableCompressors` as
  `Deflate+Delta`, `Brotli+Delta` and `Zstandard+Delta`; `new MatCompressorDelta(inner)` wraps any other byte
  compressor.
- `GetTiffBytes`: correct the documentation, it is lossless only for integer and 32/64-bit float depths with 1, 3 or 4
  channels (`Cv16F`, `Cv16BF`, `CvBool` and FP8 are converted to 8-bit), and the LZMA/Zstd schemes are not built into
  the official Windows runtime (they fall back to LZW).
- `MatCompressorPng` and `MatCompressorTiff` share the image decoding and validation logic.

# v0.4.1 (27/11/2026)

- Update Emgu.CV to 5.1.0 (OpenCV 5.1). Text rendering now uses HarfBuzz, improving complex scripts and ligatures in
  the `FontFace` text helpers.
- `DepthType.ByteCount` supports the new 8-bit float depths `Cv8FE4M3FN` and `Cv8FE4M3FNUZ`, so `CMat` can store
  matrices of these depths.
- Add `Mat.AlphaCompositeTo(background, offset, operation, premultiplied)`: composites an 8-bit BGRA overlay onto an
  8-bit 3 or 4 channel background at an offset using a Porter-Duff operator (`CvInvoke.AlphaComposite`), clipped to
  the background bounds and written in place.
- Add `Mat.GetMaskComponents(connectivity, minimumArea)`: connected regions of an 8-bit or boolean mask as managed
  `MaskComponent` values (label, bounds, area, centroid), with no native objects to dispose.
- Add `Mat.CreateOklabDistanceMask(color, tolerance)`: boolean mask of the pixels perceptually close to a color, using
  the Euclidean distance in the Oklab color space (OpenCV 5.1). Accepts BGR/BGRA `Cv8U`, `Cv16U` and `Cv32F` images.
- Add `Mat.GetTiffBytes(compression, compressionLevel)` and the `TiffCompression` enum: TIFF encoding, lossless for
  every common depth (unlike PNG), with selectable scheme (LZW, Deflate, LZMA, Zstd, ...) and level (OpenCV 5.1).
- Fix `MatCompressorPng` silently converting depths other than `Cv8U`/`Cv16U` (and channel counts other than 1, 3 or
  4) to 8-bit, which made a `CMat` using it unreadable (`InvalidDataException` on decompress). It now throws
  `NotSupportedException` for those, and `CMat` stores them raw.
- Fix `Mat.CopyTo(destination, offset, mask)` copying to the origin when an offset coordinate is `int.MinValue`.
- Bump dependencies

# v0.4.0 (15/11/2026)

- `CMat`:
  - Add `Clone(shareCompressedBytes)` and `CopyTo(dst, shareCompressedBytes)` overloads to share the read-only
    compressed buffer, reducing memory use for snapshots and undo history. Existing overloads retain deep-copy behavior;
    replacing or clearing either instance's buffer does not affect the other.
  - Add `DetachCompressedBytes`, `TryDetachCompressedBytes(expected)` and `AttachCompressedBytes` for external storage
    of buffers while preserving dimensions, depth, channels, ROI, initialization state, compression settings and hash.
    Conditional detachment rejects stale buffers, and attachment validates the restored length and content hash.
  - Preserve equality and hash codes while detached, including comparisons with resident buffers. Clones and copies
    preserve detached state and require their buffers to be restored separately.
  - Reject decompression and re-encoding until detached bytes are restored; clearing or replacing content discards the
    detached state. `CompressedLength` and `IsEmpty` continue to describe the resident buffer.
  - Add regression tests for shared-buffer isolation, compressed and raw ROI round trips, detached equality, invalid
    attachment, empty buffers, clearing detached content and concurrent detachment.
- Add .NET 11 support alongside .NET 10, targeting `net10.0` and `net11.0`.
- Enable the `MatCompressorZstd` codec on .NET 11 using the built-in `ZstandardStream` and
  `ZstandardCompressionOptions`. Support integer quality levels within the native library's limits and map
  `CompressionLevel` presets to minimum quality (`NoCompression`), quality 1 (`Fastest`), default quality (`Optimal`)
  and maximum quality (`SmallestSize`).

# v0.3.0 (10/10/2026)

- Upgrade Emgu.CV and native test runtimes to `5.0.0.6584` / OpenCV 5.
- Migrate affine and rotated-text transforms from the removed `Matrix<T>` type to `Mat` storage with span access
- Update legacy text APIs/tests to `HersheyFonts`.
- Extend `DepthType.ByteCount` for half, bfloat16, bool, uint32, uint64, and int64 storage.
- Add caller-owned `FontFace` text overloads for Unicode and custom fonts: `PutTextExtended`,
  `GetTextSizeExtended`, and `PutTextRotated`, with pixel font sizes, optional weights, wrapping,
  and whole-block alignment. Add `PutTextInRectangle` to fit and clip text within an image region.
- Add perspective helpers:
  - `CreatePerspectiveCrop` validates ordered convex corners, infers dimensions when requested, and preserves
    image depth/channels with configurable sampling and borders.
  - `OrderPerspectiveCorners` gives unordered corners a deterministic clockwise order.
  - `PerspectiveCrop` supports reusable destinations and safely handles overlapping source/destination buffers.
  - `GetPerspectiveCropSize`, `CreatePerspectiveTransform`, and `MapPerspectivePoints` support forward/inverse
    coordinate mapping, rejecting non-finite coordinates and points at infinity.
  - Add a `RotatedRect` crop overload, usable with `EmguContour.BoundsBestFit`.
  - `FindQuadrilaterals` returns ordered convex contour approximations, filtered by area and sorted largest first.
- Add boolean-mask creation and conversion helpers, plus logical `MaskAnd`, `MaskOr`, `MaskXor`, and `MaskNot`.
  Support `CvBool` and `Cv8U` input masks, including ROI views; boolean output uses canonical `0`/`1` values,
  and `ToByteMask` uses `0`/`255`.
- Optimize mask conversion and logical negation with native thresholding, use native bitwise operations for
  combinations, and fill constant boolean masks natively while preserving canonical values and ROI strides.
- Keep the Windows native test packages' NU1701 metadata warnings visible without failing the test build.
- Document resource ownership, geometry conventions, text behavior, mask formats, and OpenCV 5 migration
  considerations; add native regression tests for text, perspective geometry, buffer reuse, and masks.

# v0.2.1 (02/10/2026)

- `MatCompressor` & Subclasses:
  - `Decompress`/`DecompressAsync` now throw `ArgumentException` for an empty destination `Mat` on compressors that do
    not allocate it (everything except PNG), instead of silently returning an empty result
  - `PNG`: throw `InvalidDataException` on corrupt data (OpenCV did not report it) and decode into pre-allocated
    destinations (including ROIs) with size/type validation
  - `None`: copy with a length check instead of Emgu's unchecked `SetTo(byte[])`, and support non-continuous
    destinations
  - `Brotli`: never return an empty result for a non-empty source
  - Cache `Id`, drop redundant `async`/`await` wrappers and document the integer compression levels, the
    non-interruptible async work and the non-thread-safe `AvailableCompressors`
- `CMat`:
  - Thread safety: every property read and the `Compressor`/`CompressionLevel`/`ThresholdToCompress` setters now go
    through the lock, and `Equals`, `GetHashCode`, `ToString` and the ratio properties work on a consistent snapshot;
    the hash is computed outside the lock
  - Fix silent data loss when a compressor returns an empty result, store raw for the `None` compressor without a
    wasted compress pass, and only fall back to raw on codec errors (`ArgumentException`s now propagate)
  - Fix `SetCompressedBytes` keeping a stale `Roi` and document that it takes ownership of the array
  - Validate the decompressed size/type against the cached description (`InvalidDataException` on mismatch)
  - **Breaking (binary):** the `compressionLevel` constructor parameter is now `CompressionLevel?` and defaults to
    `MatCompressor.DefaultCompressionLevel`
- `EmguCvExtensions`:
  - `PutTextRotated`: render on a small layer instead of rotating copies of the whole image (about 30x faster on a 12MP
    image) and rotate exactly around the text origin
  - Vertical `ScanLines` walks the image row by row (cache friendly), same results and ordering
  - `GetSvgPath` formats coordinates without allocations and with the invariant culture
  - `Resize` and `ShrinkToFitPreserveAspect` never produce a zero-sized image and validate their arguments
  - `CreateLetterBox` uses area interpolation when shrinking, `RotateAdjustBounds` rounds the bounds up
  - `CreateMask` now always returns an 8-bit single-channel mask, as documented
  - `SanitizeRoiWithBehavior` treats any ROI without area as empty, `GetMemory2D(roi)` validates like `GetSpan2D(roi)`
  - `CopyAreasSmallerThan`/`CopyAreasLargerThan` add islands nested inside holes to the area of their group
  - Document `Kernel3X3Rectangle` as a shared read-only instance
- `DrawingExtensions.FactorColor` rounds instead of truncating; add `PointExtensions.FindLength(PointF, PointF)`
- `BitmapInfo`/`Mat.GetBitmapInfo()`: `RowBytes` is now the real stride (`Mat.Step`), it was the row data size, which
  was
  wrong for ROI Mats
- `EmguExtensions.Avalonia`:
  - `ToBitmap`: convert straight into the locked bitmap memory through a `Mat` view, without an intermediate Mat, and
    validate the Mat (channels, depth, color type) before allocating the bitmap
  - `ToBitmap(Type srcType)`: validate that `srcType` is an Emgu color matching the channel count, and honor it for
    4-channel Mats (e.g. `Rgba`), which were always copied as BGRA
  - Add `ToBitmap(scale, shift)`/`ToBitmapAsync` to convert 16-bit, floating point and other depths to 8-bit
  - Add `ILockedFramebuffer.ToMat()` (zero-copy view honoring the stride) and `WriteableBitmap.ToMat()` (copy)
  - **Breaking:** `WriteableBitmap.GetBitmapInfo()` no longer returns the memory address (it was only valid while the
    bitmap was locked): use the new `WithBitmapInfo` to access it inside a lock
  - **Breaking:** `ILockedFramebuffer.GetRowSpanOfBytes` excludes the row padding by default (like `GetRowSpan` and the
    core `Mat` accessors), a negative `length` now throws instead of meaning "all", and `GetPixelBytePos`/`GetPixelPos`
    validate the coordinates
  - Add `PixelPoint`/`PixelRect` overloads next to the `System.Drawing` ones, `ByteCountInt64`, checked size
    calculations, and document that the spans are only valid while the framebuffer is locked
  - Reference `CommunityToolkit.HighPerformance` explicitly and add a test suite running on the Avalonia headless
    platform

# v0.2.0 (20/09/2026)

- `EmguCvExtensions`:
  - Fix element offset calculation in `FillSpan<T>` for multi-byte types (`sizeof(T) > 1`)
  - Enable continuous memory fast-path in `ToArray` using direct span copy
  - Fix empty and non-positive dimension checks in `GetMemory2D`, `GetReadOnlyMemory2D`, `InitMat`, and
    `RoiFromBoundingRectangle`
  - Ensure unmanaged matrix disposal on exceptions in `CreateLetterBox` and `Skeletonize`
  - Fix native memory leak in `CopyAreasSmallerThan` and `CopyAreasLargerThan` by disposing contour group vectors in
    `finally` blocks
  - Optimize `GetSvgPath` by caching contour point arrays to avoid per-vertex P/Invoke calls
  - Eliminate closure allocations in `GetTextSizeExtended` and `PutTextExtended`, and allocate `linesSize` only for
    non-left text alignments
  - Align generic type constraints on `FindFirst*Pixel*` methods to `IBinaryInteger<T>, IMinMaxValue<T>`
  - Add missing `ArgumentNullException.ThrowIfNull` guards across public APIs
- `CMat`:
  - Fix ROI zero-dimension checks in `UncompressedLength`, `Compress(MatRoi)`, and `Decompress`
  - Ensure unmanaged matrix disposal on decompression failure in `RawDecompressInternal`
  - Add rollback safety in `ChangeCompressor` to preserve original dimensions and ROI if re-encoding fails
  - Improve `GetHashCode` to incorporate matrix dimensions, channels, and ROI
  - Add argument null checks on constructors and public methods
- `MatCompressor` & Subclasses:
  - Add non-continuous destination `Mat` (e.g. ROI submatrix) decompression support across `Brotli`, `Deflate`, `GZip`,
    `ZLib`, and `Zstd` compressors via row-by-row streaming into `dst.GetRowSpanOfBytes(row)`
  - Add null validation guard to `MatCompressor.DefaultCompressor` property setter
  - Add empty byte array guard and cancellation token checking in `DecompressAsync` and `CompressAsync`
  - Clean up unused constants and use `ArgumentOutOfRangeException` in `MatCompressorBrotli`
- `EmguContours`, `EmguContour`, `EmguContourFamily`:
  - Fix bug in `EmguContours.GetLargestContourArea(VectorOfVectorOfPoint, int[,])` where contour index 0 was
    unconditionally assigned
  - Fix `InvalidOperationException` in `MinSolidArea` and `MaxSolidArea` when `Families` is empty
  - Optimize `ContoursIntersectingPixels` to rasterize only the intersection bounding box `Rectangle.Intersect` instead
    of the full contour bounding box
  - Optimize `CalculateCentroidDistances` by caching contour centroid outside the inner comparison loop
  - Pre-populate `_contours` array upfront in constructors to prevent null or unassigned elements
  - Implement `IEquatable<EmguContour>` and check `IsEmpty` in `EmguContour.Centroid` to avoid P/Invoke on empty
    contours
  - Ensure unmanaged matrix disposal on exception in `ContourApproximation` and `ToVectorOfVectorOfPoint`
  - Fix `EmguContourFamily.Root` traversal to handle non-zero external depth roots safely
  - Add argument null validation guards across all public contour APIs
- Add regression tests covering all audited compressors and contour methods
- Modernize DevSkim GitHub Actions workflow (`microsoft/DevSkim-Action@v1`)
- Modernize and optimize NuGet release workflow (`.github/workflows/release.yml`) with concurrency guards, .NET package
  caching, package artifact archiving, step summaries, and release package asset attachments
- Bump dependencies

# v0.1.9 (24/07/2026)

- Add `EmguCvExtensions.CreateVector` to wrap a byte buffer in a Mat
- Refactor span/memory accessors to use shared `ResolveElementRange` helper
- Add validation methods: `ValidateByteRange`, `ValidatePixelCoordinates`, `ValidateRoi`
- Fix integer overflow in `FindLength`, `Rotate`, and polygon calculations
- Fix `CalculatePolygonRadiusFromSideLength` formula; raise minimum sides guard to 3
- Add a stack-allocation fast path in Brotli compressor for small inputs
- Replace `Marshal.Copy` with span-based copy in `SetByte`
- Fix `InitMat` array factory to dispose already-created Mats on failure
- Change `MatCompressor` equality to use Provider+Name instead of Id
- Update Avalonia to 12.1.0, DotNext to 6.4.1, and other dependencies

# v0.1.8 (27/06/2026)

- Change `Skeletonize` to use own implementation instead of `XImgProc` due to mini openCV build lacks of it

# v0.1.7 (24/06/2026)

- Add `CompressCoreAsync` and `DecompressCoreAsync` overrides to `MatCompressor`
- Add `CopyToAsync(Stream, CancellationToken)` to `EmguCvExtensions` for async raw Mat stream copies. Continuous
  matrices are written as one block; non-continuous matrices are written row by row without row padding.
- Add `Memory`/`Memory2D` accessors to `EmguCvExtensions` mirroring the span accessors: `GetMemory`,
  `GetReadOnlyMemory`, `GetRowMemory`, `GetReadOnlyRowMemory`, `GetMemory2D` (+ ROI overload), `GetReadOnlyMemory2D`
  (+ ROI overload), and their `*OfBytes` counterparts.

# v0.1.6 (31/05/2026)

- Add `EmptyRoiBehavior` enum and refactor ROI-related APIs to use ref Rectangle and explicit empty-ROI semantics.
- Add `ConstrainRoi` and `SanitizeRoiWithBehavior` helpers
- Add new SafeRoi overloads (ref-based, padding and behavior options), additional Roi/RoiFromBoundingRectangle overloads
  with padding, and RoiFromCenter adjustments.
- Rename `PutTextLineAlignment.None` to `Default` and add descriptions.
- Update `MatRoi` to use the new SafeRoi semantics.
- Update span/stream usages to ReadOnlySpan helpers
- Remove old async CopyToAsync/CropByBounds implementations.
- Tests updated to cover ConstrainRoi and to use ref SafeRoi.

# v0.1.5 (30/05/2026)

- Improve the `MatCompressor` schematic by favoring int compressionLevel instead of enum

# v0.1.4 (29/05/2026)

- Add StageKit.Primitives dependency and related using imports
- Add a new EmguExtensions.Avalonia project (bitmap helpers and Mat→WriteableBitmap converters) and a BitmapInfo record
  for describing locked framebuffer state.
- Introduce many EmguCvExtensions improvements:
  - PixelCount
  - Renamed LengthInt32/LengthInt64 → ByteCountInt32/ByteCountInt64
  - Async helpers (GetPngBytesAsync, CopyToAsync, ToBitmapAsync)
  - Additional span/stream helpers
  - Small API/exception message refinements.

# v0.1.3 (27/05/2026)

- Add/adjust EmguCvExtensions API
  - Make common color/anchor constants readonly
  - Add `Kernel3x3Rectangle`
  - Rename `InitMat(Count)` to `InitMats(Count)`
  - Add `Mat.New(Size)` and `NewZeros(Size)`, `NewFromRoiToCenter`, and default parameters for `GetSpanOfBytes` and
    `FillSpan` overloads.
  - Implement `CopyAreasSmallerThan` and `CopyAreasLargerThan` to copy contour-based regions by area.
- Add `MatRoi.Clone` convenience method.

# v0.1.2 (25/05/2026)

- Add MatCompressor `Id` and `Provider` properties, a `GetCompressorById` helper
- Add `GetSpanxxxOfBytes` methods to `EmguCvExtensions` to get spans of bytes for image data
- Rename `EmguExtensions` to `EmguCvExtensions` to not collide with the assembly name.

# v0.1.1 (29/04/2026)

- Improve `MatCompressor.Compress()` to use dotNEXT library `SparseBufferWriter`
- Improve `EmguExtensions.ScanStrides()` and `EmguExtensions.ScanLines` to use dotNEXT library `BufferWriterSlim`
- Improve `EmguExtensions.GetSvgPath()` to use dotNEXT library `BufferWriterSlim`
- Improve `EmguContours.GetEnumerator()` to get each item instead of invoking `ToArray()`
- Improve the `MatCompressorBrotli.DecompressCore()` method by using `BrotliDecoder.TryDecompress`
- Use `MemoryStream` instead of `UnmanagedMemoryStream` for `MatCompressors.DecompressCore()`

# v0.1.0 (24/04/2026)

- Initial release
