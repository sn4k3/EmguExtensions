Perform an allocation and performance optimization pass on the current file or provided code.

Focus on:
1. **Heap allocations in hot paths** — arrays, `Clone()`, `ToArray()`, `new Mat()` inside loops. Suggest reusing buffers, pre-allocating, or using stack allocation where safe.
2. **Span<T> / Memory<T> opportunities** — replace array slices, copies, or manual pointer arithmetic with `Span<T>` or `ReadOnlySpan<T>` equivalents.
3. **Redundant copies** — data copied into an intermediate buffer that could be passed directly.
4. **Unnecessary Mat allocations** — `Mat.Clone()` where `CopyTo` into a reused Mat would suffice; temporary Mats that could be hoisted out of loops.
5. **Repeated property/method calls** — properties recomputed in a loop that could be cached in a local.
6. **LINQ in hot paths** — `Select`, `Where`, `ToList`, etc. that could be replaced with a plain `for`/`foreach` + pre-allocated collection.
7. **Boxing** — value types passed as `object` or through non-generic interfaces unnecessarily.
8. **Unsafe / pointer arithmetic** — where a safe Span-based approach would be equally fast but cleaner, or where the existing unsafe code can be tightened.

For each finding state: location, what the problem is, estimated impact (high/medium/low), and the optimized code.
Do not change observable behaviour or public API signatures.
