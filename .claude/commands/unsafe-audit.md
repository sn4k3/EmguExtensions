Audit all `unsafe` blocks in the current file for correctness and safety.

Check for:
1. **Bounds overruns** — pointer arithmetic that can exceed the allocated buffer. Verify that stride, width, height, and element size are all accounted for correctly.
2. **Incorrect stride calculation** — using `Width * sizeof(T)` instead of the actual memory stride (`Step`), which may include padding. Flag anywhere `RealStep` and `Step` could be confused.
3. **Span2D / Span<T> construction** — verify `height`, `width`, and `pitch` arguments are in elements (not bytes) and match the actual memory layout.
4. **Pointer lifetime** — pointers derived from `Mat.DataPointer` or pinned arrays used after the source has been disposed or the GC could have moved it.
5. **Integer overflow in pointer arithmetic** — `int` arithmetic on large images where `width * height * channels` can exceed `int.MaxValue`; should use `long` or checked arithmetic.
6. **Unchecked casts** — `(int)` casts of values that could realistically overflow (e.g. `src.Total` which returns `IntPtr`).
7. **Missing null/empty checks before unsafe access** — dereferencing `DataPointer` on an empty or uninitialized `Mat`.
8. **`fixed` block correctness** — pinned objects accessed outside their `fixed` scope, or unnecessary `fixed` when a `Span<T>` would suffice.

For each issue state: file location, what the problem is, and the correct fix.
