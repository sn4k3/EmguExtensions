Add complete XML doc comments (`///`) to every public member in the current file or the code provided.

Rules:
- Every public class, struct, interface, enum, method, property, field, and event must have a `<summary>` tag.
- Methods and constructors: add `<param>` for every parameter and `<returns>` if the return type is not `void`.
- Add `<exception cref="...">` for every exception that can be thrown directly or documented implicitly.
- Add `<remarks>` only when the summary alone is insufficient to understand behaviour or intent.
- Do not add `<inheritdoc/>` unless the member explicitly overrides or implements a documented base.
- Do not alter any logic, signatures, or formatting outside of doc comments.
- Follow the existing doc style in the file (sentence case, period at end of summary).
