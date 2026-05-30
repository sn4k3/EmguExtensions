# v0.1.5 (30/05/2026)
- Improve the `MatCompressor` schemantic by favor int compressionLevel instead of enum

# v0.1.4 (29/05/2026)
- Add StageKit.Primitives dependency and related using imports
- Add a new EmguExtensions.Avalonia project (bitmap helpers and Mat→WriteableBitmap converters) and a BitmapInfo record for describing locked framebuffer state.
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
  - Add `Mat.New(Size)` and `NewZeros(Size)`, `NewFromRoiToCenter`, and default parameters for `GetSpanOfBytes` and `FillSpan` overloads.
  - Implement `CopyAreasSmallerThan` and `CopyAreasLargerThan` to copy contour-based regions by area. 
- Add `MatRoi.Clone` convenience method.

# v0.1.2 (25/05/2026)
- Add MatCompressor `Id` and `Provider` properties, a `GetCompressorById` helper
- Add `GetSpanxxxOfBytes` methods to `EmguCvExtensions` to get spans of bytes for image data
- Rename `EmguExtensions` to `EmguCvExtensions` to not colide with assembly name.

# v0.1.1 (29/04/2026)
- Improve `MatCompressor.Compress()` to use dotNEXT library `SparseBufferWriter`
- Improve `EmguExtensions.ScanStrides()` and `EmguExtensions.ScanLines` to use dotNEXT library `BufferWriterSlim`
- Improve `EmguExtensions.GetSvgPath()` to use dotNEXT library `BufferWriterSlim`
- Improve `EmguContours.GetEnumerator()` to get each item instead of invoking `ToArray()`
- Improve the `MatCompressorBrotli.DecompressCore()` method by using `BrotliDecoder.TryDecompress`
- Use `MemoryStream` instead of `UnmanagedMemoryStream` for `MatCompressors.DecompressCore()`

# v0.1.0 (24/04/2026)
  - Initial release