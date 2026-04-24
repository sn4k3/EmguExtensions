Perform a comprehensive review of the current file or provided code. Report only genuine problems or meaningful improvements — do not report missing XML doc comments.

---

## 1. Bugs & Correctness
- Wrong logic, off-by-one errors, incorrect conditions, data loss.
- Edge cases: empty `Mat`, zero dimensions, null inputs, single pixel, `Rectangle.Empty`, `Size.Empty`.
- Incorrect early-return guards that skip necessary state updates for callers.

## 2. Resource Leaks & IDisposable
- Any `IDisposable` (`Mat`, `MatRoi`, `CMat`, `Stream`, etc.) not wrapped in `using` or not disposed on all paths including exception paths.
- `Dispose()` called manually after code that can throw — should use `try/finally` or `using`.
- Properties or methods that allocate and return a new `IDisposable` without making ownership transfer obvious (potential caller leak trap).
- `DisposableObject` subclasses: verify `DisposeManaged()` covers all owned fields.

## 3. Nullable Correctness
- Dereferences of potentially-null values without a null check.
- Parameters, return types, or fields declared non-nullable that can actually be null.
- Missing `[NotNullWhen]`, `[MaybeNullWhen]`, `[AllowNull]`, `[NotNull]` attributes.
- `!` null-forgiving operators suppressing a real problem rather than a false positive.

## 4. Parameter Validation
- Missing `ArgumentNullException.ThrowIfNull` for reference-type parameters that must not be null.
- Missing `ArgumentOutOfRangeException` guards for numeric parameters with valid ranges.
- Missing `ArgumentException` for logically invalid combinations (e.g. mismatched dimensions, empty `Size`).
- Missing empty/uninitialized `Mat` checks before unsafe access or compression.

## 5. Performance & Allocations
- Heap allocations inside loops: `Clone()`, `ToArray()`, `new Mat()`, LINQ.
- Redundant copies or double-assignment (e.g. assigning a property then immediately reassigning it).
- Repeated property/method calls in a loop that could be cached in a local.
- `Span<T>` / `Memory<T>` opportunities replacing manual pointer arithmetic or array slices.
- Unnecessary intermediate `Mat` allocations that could be hoisted or reused via `CopyTo`.

## 6. Unsafe Code
- Pointer arithmetic that can exceed the allocated buffer (bounds overruns).
- Confusion between `Step` (stride with padding) and `RealStep` (actual data width) in index calculations.
- `Span2D` construction: verify `height`, `width`, and `pitch` are in **elements** (not bytes).
- Integer overflow in pointer/index arithmetic on large images (`int` vs `long`).
- Unchecked `(int)` casts of values that could overflow (e.g. `src.Total` which returns `IntPtr`).
- Dereferencing `DataPointer` on an empty or uninitialized `Mat`.

## 7. API Design
- Surprising behaviour for callers: mutable state exposed unintentionally, properties that allocate without signalling it.
- Naming inconsistencies with the rest of the extension methods.
- Overload gaps or parameter ordering that diverges from the established convention.
- `GetHashCode` / `Equals` contracts broken by mutable state.

---

For each finding state: **location** (method/property name), **what is wrong**, and **the fix**.
Group findings by section. Omit any section with no findings.
