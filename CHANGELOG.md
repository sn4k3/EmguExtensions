# v0.1.1 (29/04/2026)
- Improve `MatCompressor.Compress()` to use dotNEXT library `SparseBufferWriter`
- Improve `EmguExtensions.ScanStrides()` and `EmguExtensions.ScanLines` to use dotNEXT library `BufferWriterSlim`
- Improve `EmguExtensions.GetSvgPath()` to use dotNEXT library `BufferWriterSlim`
- Improve `EmguContours.GetEnumerator()` to get each item instead of invoking `ToArray()`
- Improve the `MatCompressorBrotli.DecompressCore()` method by using `BrotliDecoder.TryDecompress`
- Use `MemoryStream` instead of `UnmanagedMemoryStream` for `MatCompressors.DecompressCore()`

# v0.1.0 (24/04/2026)
  - Initial release