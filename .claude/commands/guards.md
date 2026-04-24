Add parameter validation guards to all public methods and constructors in the current file or provided code.

Rules:
- Throw `ArgumentNullException.ThrowIfNull(param)` for any reference-type parameter that must not be null.
- Throw `ArgumentOutOfRangeException.ThrowIfNegative`, `ThrowIfZero`, `ThrowIfLessThan`, `ThrowIfGreaterThan` (or equivalent) for numeric parameters with obvious valid ranges.
- Throw `ArgumentException` for logically invalid combinations (e.g. empty Size, mismatched dimensions).
- For Mat parameters: guard against `mat.IsEmpty` where the method cannot meaningfully operate on an empty Mat.
- Place all guards at the very top of the method body, before any other logic.
- Use the modern static throw helpers (`ArgumentNullException.ThrowIfNull`, etc.) introduced in .NET 6+ — do not write `if (...) throw new ...` manually unless no static helper exists.
- Do not add guards to private methods unless they are called from multiple sites with untrusted input.
- Do not change any logic beyond adding the guard statements.
