Audit the current file for resource leaks and IDisposable correctness.

Check for:
- Any `IDisposable` (Mat, MatRoi, CMat, Stream, MemoryStream, etc.) that is created but not wrapped in `using` or not disposed on every code path, including exception paths.
- `try/finally` blocks where `Dispose()` is called manually but could throw or be skipped.
- Methods that return a newly allocated `Mat` or other `IDisposable` — verify the caller contract is clear (ownership transfer vs. borrowed reference).
- `using var` declarations that go out of scope while the resource is still referenced by something alive.
- Properties or methods whose name does not hint at allocation (e.g. plain getters) but actually create and return a new `IDisposable` — flag these as potential caller leak traps.
- Subclasses of `DisposableObject`: verify `DisposeManaged()` disposes every owned `IDisposable` field, and `DisposeUnmanaged()` is overridden when there are unmanaged handles.

For each issue state: location, what leaks, and the fix.
