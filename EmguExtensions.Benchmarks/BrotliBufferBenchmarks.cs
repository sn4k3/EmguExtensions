using System.IO.Compression;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Order;
using DotNext.Buffers;
using DotNext.IO;

namespace EmguExtensions.Benchmarks;

/// <summary>
/// Compares Brotli input streams used for decompression and output buffers used for compression.
/// </summary>
[MemoryDiagnoser]
[Orderer(SummaryOrderPolicy.FastestToSlowest)]
[GroupBenchmarksBy(BenchmarkLogicalGroupRule.ByCategory)]
public class BrotliBufferBenchmarks
{
    [Params(1920 * 1080)]
    public int ByteCount { get; set; }

    [Params(CompressionLevel.Fastest, CompressionLevel.Optimal)]
    public CompressionLevel Level { get; set; }

    private byte[] _sourceBytes = null!;
    private byte[] _compressedBytes = null!;
    private byte[] _decompressedBytes = null!;

    [GlobalSetup]
    public void Setup()
    {
        _sourceBytes = GC.AllocateUninitializedArray<byte>(ByteCount);
        for (int i = 0; i < _sourceBytes.Length; i++)
        {
            _sourceBytes[i] = (byte)((i * 31 + i / 97) % 256);
        }

        _compressedBytes = CompressWithSparseBufferWriterCore();
        _decompressedBytes = GC.AllocateUninitializedArray<byte>(ByteCount);
    }

    [Benchmark(Baseline = true, Description = "Decompress: MemoryStream")]
    [BenchmarkCategory("Decompress")]
    public byte DecompressWithMemoryStream()
    {
        using var compressedStream = new MemoryStream(_compressedBytes, writable: false);
        using var brotliStream = new BrotliStream(compressedStream, CompressionMode.Decompress);
        brotliStream.ReadExactly(_decompressedBytes);
        return _decompressedBytes[^1];
    }

    [Benchmark(Description = "Decompress: UnmanagedMemoryStream")]
    [BenchmarkCategory("Decompress")]
    public unsafe byte DecompressWithUnmanagedMemoryStream()
    {
        fixed (byte* pBuffer = _compressedBytes)
        {
            using var compressedStream = new UnmanagedMemoryStream(pBuffer, _compressedBytes.Length);
            using var brotliStream = new BrotliStream(compressedStream, CompressionMode.Decompress);
            brotliStream.ReadExactly(_decompressedBytes);
        }

        return _decompressedBytes[^1];
    }

    [Benchmark(Description = "Decompress: BrotliDecoder")]
    [BenchmarkCategory("Decompress")]
    public int DecompressWithBrotliDecoder()
    {
        if (!BrotliDecoder.TryDecompress(_compressedBytes, _decompressedBytes, out var bytesWritten))
        {
            throw new InvalidDataException("Brotli decompression failed.");
        }

        if (bytesWritten != _decompressedBytes.Length)
        {
            throw new InvalidDataException("Brotli decompressed size does not match the destination buffer.");
        }

        return bytesWritten;
    }

    [Benchmark(Baseline = true, Description = "Compress: MemoryStream")]
    [BenchmarkCategory("Compress")]
    public byte[] CompressWithMemoryStream()
    {
        using var compressedStream = new MemoryStream();
        using (var brotliStream = new BrotliStream(compressedStream, Level, leaveOpen: true))
        {
            brotliStream.Write(_sourceBytes);
        }

        return compressedStream.ToArray();
    }

    [Benchmark(Description = "Compress: SparseBufferWriter")]
    [BenchmarkCategory("Compress")]
    public byte[] CompressWithSparseBufferWriter()
    {
        return CompressWithSparseBufferWriterCore();
    }

    [Benchmark(Description = "Compress: BufferWriterSlim")]
    [BenchmarkCategory("Compress")]
    public byte[] CompressWithBufferWriterSlim()
    {
        var maxCompressedLength = BrotliEncoder.GetMaxCompressedLength(_sourceBytes.Length);
        var buffer = new BufferWriterSlim<byte>(maxCompressedLength);
        try
        {
            if (!BrotliEncoder.TryCompress(
                    _sourceBytes,
                    buffer.GetSpan(maxCompressedLength),
                    out var bytesWritten,
                    GetBrotliQuality(Level),
                    GetBrotliWindow(Level)))
            {
                throw new InvalidDataException("Brotli compression failed.");
            }

            buffer.Advance(bytesWritten);
            return ToArray(buffer.WrittenSpan);
        }
        finally
        {
            buffer.Dispose();
        }
    }

    private byte[] CompressWithSparseBufferWriterCore()
    {
        using var buffer = new SparseBufferWriter<byte>(Math.Min(64 * 1024, _sourceBytes.Length), SparseBufferGrowth.Linear);
        using (var brotliStream = new BrotliStream(Stream.Create(buffer, true), Level))
        {
            brotliStream.Write(_sourceBytes);
        }

        return buffer.ToArray();
    }

    private static int GetBrotliQuality(CompressionLevel level)
    {
        return level switch
        {
            CompressionLevel.NoCompression => 0,
            CompressionLevel.Fastest => 1,
            CompressionLevel.SmallestSize => 11,
            _ => 5
        };
    }

    private static int GetBrotliWindow(CompressionLevel level)
    {
        return level == CompressionLevel.NoCompression ? 10 : 22;
    }

    private static byte[] ToArray(ReadOnlySpan<byte> span)
    {
        if (span.IsEmpty) return [];

        var result = GC.AllocateUninitializedArray<byte>(span.Length);
        span.CopyTo(result);
        return result;
    }
}
