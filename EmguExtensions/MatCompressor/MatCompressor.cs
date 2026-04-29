/*
*   MIT License
*
*   Copyright (c) 2026 Tiago Conceição
*
*   Permission is hereby granted, free of charge, to any person obtaining a copy
*   of this software and associated documentation files (the "Software"), to deal
*   in the Software without restriction, including without limitation the rights
*   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
*   copies of the Software, and to permit persons to whom the Software is
*   furnished to do so, subject to the following conditions:
*
*   The above copyright notice and this permission notice shall be included in all
*   copies or substantial portions of the Software.
*
*   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
*   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
*   SOFTWARE.
*/

using System.Collections.ObjectModel;
using System.IO.Compression;
using DotNext.Buffers;
using DotNext.IO;
using Emgu.CV;

namespace EmguExtensions;

/// <summary>
/// Provides an abstract base class for implementing matrix compression and decompression algorithms for Mat objects.
/// </summary>
/// <remarks>Inherit from this class to create custom compressors for Mat data. Implementations must provide logic
/// for compressing and decompressing matrices, as well as a unique compressor name. This class also provides
/// asynchronous methods for compression and decompression, which offload work to background threads.</remarks>
public abstract class MatCompressor
{
    /// <summary>
    /// Gets a collection of available material compressors supported by the system.
    /// </summary>
    /// <remarks>The collection includes built-in compressors such as None, PNG, Deflate, GZip, ZLib, and Brotli.
    /// The collection is read-only and can be used to enumerate or select a compressor for material processing
    /// operations.</remarks>
    public static ObservableCollection<MatCompressor> AvailableCompressors { get; } = [
        MatCompressorNone.Instance,
        MatCompressorPng.Instance,
        MatCompressorDeflate.Instance,
        MatCompressorGZip.Instance,
        MatCompressorZLib.Instance,
        MatCompressorBrotli.Instance,
#if NET11_0_OR_GREATER
        MatCompressorZstd.Instance,
#endif
    ];

    /// <summary>
    /// Gets or sets the default compressor to be used for material compression operations when no specific compressor is specified, also used for CMat.
    /// </summary>
    public static MatCompressor DefaultCompressor { get; set; } =
#if NET11_0_OR_GREATER
        MatCompressorZstd.Instance;
#else
        MatCompressorBrotli.Instance;
#endif

    /// <summary>
    /// Gets or sets the default compression level to be used for material compression operations when no specific level is specified, also used for CMat.
    /// </summary>
    public static CompressionLevel DefaultCompressionLevel { get; set; } = CompressionLevel.Optimal;

    /// <summary>
    /// Gets or sets the default chunk size (in bytes) to be used for compressing mats in chunks. This can help manage memory usage and improve performance when dealing with large matrices.
    /// </summary>
    public static int DefaultBufferChunkSize { get; set; } = 64 * 1024;

    /// <summary>
    /// Gets a compressor by its name from the collection of available compressors.
    /// </summary>
    /// <param name="name">The name of the compressor.</param>
    /// <returns>The compressor with the specified name, or null if not found.</returns>
    public static MatCompressor? GetCompressorByName(string name)
    {
        if (string.IsNullOrWhiteSpace(name)) return null;
        return AvailableCompressors.FirstOrDefault(compressor =>
            compressor.Name.Equals(name, StringComparison.OrdinalIgnoreCase));
    }

    /// <summary>
    /// Gets the name of the compressor
    /// </summary>
    public abstract string Name { get; }

    /// <summary>
    /// Gets the growth strategy used by stream-based compressors.
    /// </summary>
    protected virtual SparseBufferGrowth BufferGrowth => SparseBufferGrowth.Linear;

    /// <summary>
    /// Determines the optimal buffer chunk size for compressing the given <see cref="Mat"/>. This method can be overridden by derived classes to provide custom logic for determining the chunk size based on the characteristics of the matrix, such as its dimensions, data type, or memory usage. By default, it returns the minimum of a predefined default chunk size and the total byte length of the matrix data to ensure efficient memory usage during compression.
    /// </summary>
    /// <param name="mat">The <see cref="Mat"/> for which to determine the optimal buffer chunk size.</param>
    /// <returns>The optimal buffer chunk size in bytes.</returns>
    protected virtual int GetOptimalBufferChunkSize(Mat mat)
    {
        return Math.Min(DefaultBufferChunkSize, mat.LengthInt32);
    }

    /// <summary>
    /// Creates a sparse byte buffer for stream-based compressors.
    /// </summary>
    /// <param name="mat">The <see cref="Mat"/> to compress.</param>
    /// <returns>A sparse byte buffer sized for the source matrix.</returns>
    protected SparseBufferWriter<byte> CreateCompressionBuffer(Mat mat)
    {
        return new SparseBufferWriter<byte>(GetOptimalBufferChunkSize(mat), BufferGrowth);
    }

    /// <summary>
    /// Creates a writable stream over a sparse compression buffer.
    /// </summary>
    /// <param name="buffer">The sparse byte buffer.</param>
    /// <returns>A writable stream backed by <paramref name="buffer"/>.</returns>
    protected static Stream CreateCompressionStream(SparseBufferWriter<byte> buffer)
    {
        return Stream.Create(buffer, true);
    }

    /// <summary>
    /// Compresses the <see cref="Mat"/> into a byte array using the default compression level (<see cref="DefaultCompressionLevel"/>).
    /// </summary>
    /// <param name="src">The source <see cref="Mat"/> to compress.</param>
    /// <returns>A byte array containing the compressed data.</returns>
    public virtual byte[] Compress(Mat src)
    {
        return Compress(src, DefaultCompressionLevel);
    }

    /// <summary>
    /// Compresses the <see cref="Mat"/> into a byte array.
    /// </summary>
    /// <param name="src"></param>
    /// <param name="compressionLevel"></param>
    /// <returns></returns>
    public virtual byte[] Compress(Mat src, int compressionLevel)
    {
        return Compress(src, CompressionExtensions.GetCompressionLevel(compressionLevel));
    }

    /// <summary>
    /// Compresses the <see cref="Mat"/> into a byte array.
    /// </summary>
    /// <param name="src"></param>
    /// <param name="compressionLevel"></param>
    /// <returns></returns>
    public byte[] Compress(Mat src, CompressionLevel compressionLevel)
    {
        if (src.IsEmpty) return [];
        return CompressCore(src, compressionLevel);
    }

    /// <summary>
    /// Performs the actual compression. Only called when <paramref name="src"/> is non-empty.
    /// </summary>
    /// <param name="src"></param>
    /// <param name="compressionLevel"></param>
    /// <returns></returns>
    protected abstract byte[] CompressCore(Mat src, CompressionLevel compressionLevel);

    /// <summary>
    /// Decompresses the <see cref="Mat"/> from a byte array.
    /// </summary>
    /// <param name="compressedBytes"></param>
    /// <param name="dst"></param>
    public void Decompress(byte[] compressedBytes, Mat dst)
    {
        if (compressedBytes.Length == 0) return;
        DecompressCore(compressedBytes, dst);
    }

    /// <summary>
    /// Performs the actual decompression. Only called when <paramref name="compressedBytes"/> is non-empty.
    /// </summary>
    /// <param name="compressedBytes"></param>
    /// <param name="dst"></param>
    protected abstract void DecompressCore(byte[] compressedBytes, Mat dst);

    /// <summary>
    /// Compresses the <see cref="Mat"/> into a byte array asynchronously.
    /// </summary>
    /// <param name="src">The source <see cref="Mat"/> to compress.</param>
    /// <param name="cancellationToken">A token to monitor for cancellation requests.</param>
    /// <returns>A task that represents the asynchronous operation. The task result contains a byte array with the compressed data.</returns>
    public Task<byte[]> CompressAsync(Mat src, CancellationToken cancellationToken = default)
    {
        return Task.Run(() => Compress(src), cancellationToken);
    }

    /// <summary>
    /// Compresses the <see cref="Mat"/> into a byte array asynchronously.
    /// </summary>
    public Task<byte[]> CompressAsync(Mat src, int compressionLevel, CancellationToken cancellationToken = default)
    {
        return Task.Run(() => Compress(src, compressionLevel), cancellationToken);
    }

    /// <summary>
    /// Compresses the <see cref="Mat"/> into a byte array asynchronously.
    /// </summary>
    public Task<byte[]> CompressAsync(Mat src, CompressionLevel compressionLevel, CancellationToken cancellationToken = default)
    {
        return Task.Run(() => Compress(src, compressionLevel), cancellationToken);
    }

    /// <summary>
    /// Decompresses the <see cref="Mat"/> from a byte array asynchronously.
    /// </summary>
    public Task DecompressAsync(byte[] compressedBytes, Mat dst, CancellationToken cancellationToken = default)
    {
        return Task.Run(() => Decompress(compressedBytes, dst), cancellationToken);
    }

    /// <inheritdoc />
    public override string ToString()
    {
        return $"Compressor: {Name}";
    }
}

/*
#region LZ4
public sealed class MatCompressorLz4 : MatCompressor
{
    public static readonly MatCompressorLz4 Instance = new();

    private MatCompressorLz4() { }

    private static LZ4Level GetLZ4Level() => CoreSettings.DefaultLayerCompressionLevel switch
    {
        LayerCompressionLevel.Lowest => LZ4Level.L00_FAST,
        LayerCompressionLevel.Highest => LZ4Level.L12_MAX,
        _ => LZ4Level.L10_OPT
    };

    public override byte[] Compress(Mat src, CompressionLevel compressionLevel, object? argument = null)
    {
        using var compressedStream = StreamExtensions.RecyclableMemoryStreamManager.GetStream();
        using (var lz4Stream = LZ4Stream.Encode(compressedStream, GetLZ4Level(), leaveOpen: true))
        {
            CompressToStream(src, lz4Stream, argument);
        }

        return compressedStream.TryGetBuffer(out var buffer)
            ? buffer.ToArray()
            : compressedStream.ToArray();
    }

    public override void Decompress(byte[] compressedBytes, Mat dst, object? argument = null)
    {
        unsafe
        {
            fixed (byte* pBuffer = compressedBytes)
            {
                using var compressedStream = new UnmanagedMemoryStream(pBuffer, compressedBytes.Length);
                using var lz4Stream = LZ4Stream.Decode(compressedStream, leaveOpen: true);
                lz4Stream.ReadExactly(dst.GetDataByteSpan());
            }
        }
    }

    public override string ToString() => "LZ4";
}
#endregion
*/
