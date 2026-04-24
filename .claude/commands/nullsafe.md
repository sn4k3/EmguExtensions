Audit the current file for nullable reference type correctness. The project has `<Nullable>enable</Nullable>`.

Check for:
- Dereferences of potentially-null values without a prior null check or null-forgiving operator.
- Parameters that should be `T?` but are declared `T`, or vice versa.
- Return types that can return `null` but are declared non-nullable.
- Missing `[NotNullWhen]`, `[MaybeNullWhen]`, `[AllowNull]`, or `[NotNull]` attributes where they would silence a false-positive or document a contract.
- `!` (null-forgiving) operators used to suppress warnings instead of fixing the root cause — flag these unless there is a genuine reason.
- Properties or fields assigned in a constructor path that the compiler cannot prove are non-null.

For each issue state: location, what the problem is, and the correct fix or annotation.
Do not change any logic — only nullability annotations and guard checks.
