# v0.1.10 (20/09/2026)

- `EmguCvExtensions`:
  - Fix element offset calculation in `FillSpan<T>` for multi-byte types (`sizeof(T) > 1`)
  - Enable continuous memory fast-path in `ToArray` using direct span copy
  - Fix empty and non-positive dimension checks in `GetMemory2D`, `GetReadOnlyMemory2D`, `InitMat`, and `RoiFromBoundingRectangle`
  - Ensure unmanaged matrix disposal on exceptions in `CreateLetterBox` and `Skeletonize`
  - Fix native memory leak in `CopyAreasSmallerThan` and `CopyAreasLargerThan` by disposing contour group vectors in `finally` blocks
  - Optimize `GetSvgPath` by caching contour point arrays to avoid per-vertex P/Invoke calls
  - Eliminate closure allocations in `GetTextSizeExtended` and `PutTextExtended`, and allocate `linesSize` only for non-left text alignments
  - Align generic type constraints on `FindFirst*Pixel*` methods to `IBinaryInteger<T>, IMinMaxValue<T>`
  - Add missing `ArgumentNullException.ThrowIfNull` guards across public APIs
- `CMat`:
  - Fix ROI zero-dimension checks in `UncompressedLength`, `Compress(MatRoi)`, and `Decompress`
  - Ensure unmanaged matrix disposal on decompression failure in `RawDecompressInternal`
  - Add rollback safety in `ChangeCompressor` to preserve original dimensions and ROI if re-encoding fails
  - Improve `GetHashCode` to incorporate matrix dimensions, channels, and ROI
  - Add argument null checks on constructors and public methods
- `MatCompressor` & Subclasses:
  - Add non-continuous destination `Mat` (e.g. ROI submatrix) decompression support across `Brotli`, `Deflate`, `GZip`, `ZLib`, and `Zstd` compressors via row-by-row streaming into `dst.GetRowSpanOfBytes(row)`
  - Add null validation guard to `MatCompressor.DefaultCompressor` property setter
  - Add empty byte array guard and cancellation token checking in `DecompressAsync` and `CompressAsync`
  - Clean up unused constants and use `ArgumentOutOfRangeException` in `MatCompressorBrotli`
- `EmguContours`, `EmguContour`, `EmguContourFamily`:
  - Fix bug in `EmguContours.GetLargestContourArea(VectorOfVectorOfPoint, int[,])` where contour index 0 was unconditionally assigned
  - Fix `InvalidOperationException` in `MinSolidArea` and `MaxSolidArea` when `Families` is empty
  - Optimize `ContoursIntersectingPixels` to rasterize only the intersection bounding box `Rectangle.Intersect` instead of the full contour bounding box
  - Optimize `CalculateCentroidDistances` by caching contour centroid outside the inner comparison loop
  - Pre-populate `_contours` array upfront in constructors to prevent null or unassigned elements
  - Implement `IEquatable<EmguContour>` and check `IsEmpty` in `EmguContour.Centroid` to avoid P/Invoke on empty contours
  - Ensure unmanaged matrix disposal on exception in `ContourApproximation` and `ToVectorOfVectorOfPoint`
  - Fix `EmguContourFamily.Root` traversal to handle non-zero external depth roots safely
  - Add argument null validation guards across all public contour APIs
- Add regression tests covering all audited compressors and contour methods
- Modernize DevSkim GitHub Actions workflow (`microsoft/DevSkim-Action@v1`)
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

