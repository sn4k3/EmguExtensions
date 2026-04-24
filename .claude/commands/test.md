Generate thorough unit tests for the method, class, or file provided.

Rules:
- Use xUnit (`[Fact]`, `[Theory]`, `[InlineData]`) as the test framework.
- Test method naming: `MethodName_Scenario_ExpectedResult` (e.g. `Compress_EmptyMat_SetsIsInitializedTrue`).
- Cover: happy path, edge cases (empty Mat, zero dimensions, single pixel, max values), and expected exceptions for invalid input.
- For Mat-based tests: always dispose Mats under `using`. Create test Mats with known pixel data to assert exact output.
- For compression tests: verify round-trip fidelity (compress → decompress → pixel-equal to original).
- Assert with `Assert.Equal`, `Assert.True`, `Assert.Throws<T>`, etc. — no `Assert.Pass` or loose checks.
- Do not use mocking frameworks unless the dependency is an interface. Prefer real objects.
- Group tests for the same class in a single `public class CMatTests { }` (or equivalent) file.
- Each test must be fully self-contained — no shared mutable state between tests.
- Note any cases that are difficult to test due to the current API design (e.g. properties that allocate without signalling ownership).
