using System.Reflection;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Running;
using EmguExtensions.Benchmarks;
#if NET11_0_OR_GREATER
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Toolchains.InProcess.Emit;
#endif

// Rider can launch from the redirected output folder, outside the source tree.
// BenchmarkDotNet's out-of-process generator searches for the project from the working directory.
var projectDirectory = typeof(MatCompressorBenchmarks)
    .Assembly.GetCustomAttributes<AssemblyMetadataAttribute>()
    .Single(attribute => attribute.Key == "BenchmarkProjectDirectory")
    .Value
    ?? throw new InvalidOperationException("The benchmark project directory is missing.");
Directory.SetCurrentDirectory(projectDirectory);

var compressorConfig = DefaultConfig.Instance.AddFilter(new RedundantCompressorLevelFilter());
#if NET11_0_OR_GREATER
// BenchmarkDotNet 0.15.8 cannot validate the .NET 11 SDK for out-of-process jobs.
compressorConfig = compressorConfig.AddJob(
    Job.Default.WithToolchain(InProcessEmitToolchain.Instance).AsMutator()
);
#endif
BenchmarkRunner.Run<MatCompressorBenchmarks>(compressorConfig, args);

//BenchmarkRunner.Run<MatToArrayBenchmarks>(null, args);
//BenchmarkRunner.Run<BrotliBufferBenchmarks>(null, args);
