using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Running;
using EmguExtensions.Benchmarks;


var compressorConfig = DefaultConfig.Instance
    .AddFilter(new RedundantCompressorLevelFilter());
BenchmarkRunner.Run<MatCompressorBenchmarks>(compressorConfig, args);

//BenchmarkRunner.Run<MatToArrayBenchmarks>(null, args);