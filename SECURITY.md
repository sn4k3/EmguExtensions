# Security Policy

The EmguExtensions maintainers take security seriously. This document describes how to report vulnerabilities and what you can expect in response.

## Supported Versions

Only the latest released version on [NuGet](https://www.nuget.org/packages/EmguExtensions) receives security fixes. Older versions will not be patched — upgrade to the latest release to receive fixes.

| Version | Supported          |
| ------- | ------------------ |
| Latest  | :white_check_mark: |
| Older   | :x:                |

While EmguExtensions is pre-1.0 (0.x), any release may contain breaking changes. Security patches will ship as new minor or patch versions of the current 0.x line.

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues, discussions, or pull requests.**

Instead, report them privately through one of the following channels:

1. **GitHub Private Vulnerability Reporting** (preferred):
   Use the [Report a vulnerability](https://github.com/sn4k3/EmguExtensions/security/advisories/new) button on the repository's Security tab. This creates a private advisory visible only to maintainers.

2. **Email**:
   Send a report to the maintainer at the address listed on the [GitHub profile](https://github.com/sn4k3). Use a clear subject like `[EmguExtensions security] <short description>`.

### What to include

To help us triage quickly, please include as much of the following as you can:

- A description of the vulnerability and its impact
- The affected version(s) of EmguExtensions
- Steps to reproduce (minimal code sample, input data, or proof-of-concept)
- Any relevant stack traces, logs, or crash dumps
- Your assessment of severity (e.g. remote code execution, denial of service, information disclosure)
- Whether the issue has been disclosed elsewhere

If the issue relates to a dependency (for example Emgu.CV, OpenCV, or a compression library), please say so — we may need to forward the report to the upstream maintainer.

## Response Process

You can expect the following timeline for a reported vulnerability:

| Stage                          | Target                         |
| ------------------------------ | ------------------------------ |
| Initial acknowledgement        | Within 7 days                  |
| Triage and severity assessment | Within 14 days                 |
| Fix or mitigation plan         | Within 30 days where feasible  |
| Coordinated disclosure         | Negotiated with the reporter   |

This is a volunteer-maintained open-source project, so timelines are best-effort. Complex issues, upstream dependencies, or reports received during periods of reduced availability may take longer.

We will:

- Acknowledge your report and work with you to confirm the issue.
- Keep you informed of progress toward a fix.
- Credit you in the release notes and security advisory if you wish (anonymous reports are also welcome).
- Coordinate public disclosure with you once a fix is available.

## Scope

In scope:

- Vulnerabilities in the `EmguExtensions` library source code (this repository) — for example memory-safety issues in `unsafe` blocks, incorrect bounds handling in ROI or span accessors, use-after-free in disposal logic, or decompression routines that can be driven into unbounded allocation or crashes by crafted input.
- Security issues in the NuGet package metadata, signing, or build pipeline (GitHub Actions workflows in this repository).

Out of scope:

- Vulnerabilities in [Emgu.CV](https://github.com/emgucv/emgucv) or the underlying OpenCV native libraries — please report those upstream.
- Vulnerabilities in .NET runtime packages (`System.IO.Hashing`, `Microsoft.IO.RecyclableMemoryStream`, `CommunityToolkit.HighPerformance`, etc.) — please report those to their respective maintainers.
- Issues that require an attacker to already have arbitrary code execution on the host.
- Build-time or development-time issues that do not affect consumers of the published package.

If you are unsure whether something is in scope, report it anyway and we will route it appropriately.

## Security Considerations for Consumers

A few notes for applications using EmguExtensions:

- **Untrusted input to decompressors.** `CMat.Decompress()` and the `MatCompressor` family decode arbitrary byte buffers. Feeding untrusted input to these APIs is supported, but as with any decompression library, crafted payloads can cause high CPU or memory usage. Apply input size limits and timeouts at your application's trust boundary.
- **`unsafe` / pointer APIs.** Several APIs (`BytePointer`, `GetUnmanagedMemoryStream`, `GCSafeHandle`, span accessors backed by unmanaged memory) expose pointers into `Mat` buffers. The caller is responsible for keeping the source `Mat` alive and not disposed for the duration of the pointer's use. Misuse is a correctness issue, not a vulnerability in the library.
- **Thread-safety guarantees.** Only `CMat` is documented as thread-safe for concurrent `Compress`/`Decompress`/`CopyTo`. Other types follow standard .NET instance-member-not-thread-safe conventions.
- **NativeAOT / trimming.** This library is not currently validated for NativeAOT or aggressive trimming (`Emgu.CV` does not declare trim or AOT compatibility). Do not assume trim safety.

## Acknowledgements

We are grateful to the security community for responsible disclosure. Reporters who follow this policy will be credited in the affected release's notes unless they request otherwise.
