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
public abstract class MatCompressor : IEquatable<MatCompressor>
{
    #region Formatters

    /// <inheritdoc />
    public override string ToString()
    {
        return $"Provider: {Provider}, Compressor: {Name}";
    }

    #endregion

    #region Static Properties

    /// <summary>
    /// Gets a collection of available material compressors supported by the system.
    /// </summary>
    /// <remarks>The collection includes built-in compressors such as None, PNG, Deflate, GZip, ZLib, and Brotli.
    /// The collection is read-only and can be used to enumerate or select a compressor for material processing
    /// operations.</remarks>
    public static ObservableCollection<MatCompressor> AvailableCompressors { get; } =
    [
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
    public static int DefaultBufferChunkSize
    {
        get => field;
        set => field = Math.Max(64, value);
    } = 64 * 1024;

    #endregion

    #region Properties

    /// <summary>
    /// Gets a unique identifier for the compressor, combining the provider and name properties.
    /// This can be useful for distinguishing between different compressors, especially if multiple compressors share the same name but come from different providers or libraries.
    /// </summary>
    public string Id => $"{Provider}#{Name}";

    /// <summary>
    /// Gets the provider or library used by this compressor, if applicable.
    /// This can be useful for informational purposes or to identify the underlying implementation of the compressor (e.g., "Brotli", "ZLib", "LZ4", etc.).
    /// By default, it returns ".NET" to indicate that the compressor is implemented using .NET's built-in compression libraries, but derived classes can override this property to specify a different provider if they use an external library or custom implementation.
    /// </summary>
    public virtual string Provider => ".NET";

    /// <summary>
    /// Gets the name of the compressor. This should be a unique name that identifies the specific compression algorithm or method implemented by this compressor (e.g., "Brotli", "ZLib", "LZ4", etc.).
    /// The name is used in the <see cref="Id"/> property to create a unique identifier for the compressor and can also be used for display purposes when listing available compressors or providing information about the compressor being used.
    /// </summary>
    public abstract string Name { get; }

    /// <summary>
    /// Gets the minimum compression level supported by this compressor. This can be used to validate user input or to provide information about the range of valid compression levels for this compressor.<br/>
    /// By default, it returns 0, which typically represents the lowest level of compression (or no compression) in many compression libraries, but derived classes can override this property to specify a different minimum level if their compression algorithm uses a different scale or range for compression levels.
    /// </summary>
    public virtual int MinimumCompressionLevel => 0;

    /// <summary>
    /// Gets the maximum compression level supported by this compressor. This can be used to validate user input or to provide information about the range of valid compression levels for this compressor.<br/>
    /// By default, it returns <see cref="CompressionLevel.SmallestSize"/> which is <c>3</c>, which typically represents the highest level of compression in many compression libraries, but derived classes can override this property to specify a different maximum level if their compression algorithm uses a different scale or range for compression levels.
    /// </summary>
    public virtual int MaximumCompressionLevel => 3;

    /// <summary>
    /// Gets the growth strategy used by stream-based compressors.
    /// </summary>
    protected virtual SparseBufferGrowth BufferGrowth => SparseBufferGrowth.Linear;

    #endregion

    #region Utility Methods

    /// <summary>
    /// Converts a <see cref="CompressionLevel"/> enum value to the corresponding integer compression level used by the compressor.
    /// This method can be overridden by derived classes to provide custom logic for mapping the standard <see cref="CompressionLevel"/> values to the specific integer levels expected by the underlying compression algorithm.<br/>
    /// By default, it simply casts the <see cref="CompressionLevel"/> to an integer, which works for many compressors that use the standard .NET compression levels, but some compressors may require a different mapping or scaling of these values.
    /// </summary>
    /// <param name="compressionLevel">The <see cref="CompressionLevel"/> value to convert.</param>
    /// <returns>The corresponding integer compression level.</returns>
    protected virtual int GetCompressionLevel(CompressionLevel compressionLevel)
    {
        return (int)compressionLevel;
    }

    /// <summary>
    /// Determines the optimal buffer chunk size for compressing the given <see cref="Mat"/>. This method can be overridden by derived classes to provide custom logic for determining the chunk size based on the characteristics of the matrix, such as its dimensions, data type, or memory usage. By default, it returns the minimum of a predefined default chunk size and the total byte length of the matrix data to ensure efficient memory usage during compression.
    /// </summary>
    /// <param name="mat">The <see cref="Mat"/> for which to determine the optimal buffer chunk size.</param>
    /// <returns>The optimal buffer chunk size in bytes.</returns>
    protected virtual int GetOptimalBufferChunkSize(Mat mat)
    {
        return Math.Min(DefaultBufferChunkSize, mat.ByteCountInt32);
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
    /// Validates that the provided compression level is within the supported range for this compressor.<br/>
    /// If the compression level is outside the valid range, an <see cref="ArgumentOutOfRangeException"/> is thrown with a descriptive error message indicating the valid range and the compressor details.
    /// </summary>
    /// <param name="compressionLevel">The compression level to validate.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown if the compression level is outside the valid range.</exception>
    protected void ValidateCompressionLevel(int compressionLevel)
    {
        if (MinimumCompressionLevel == MaximumCompressionLevel)
            return; // No validation needed if there's only one valid level. Often this is None compression.
        if (compressionLevel < MinimumCompressionLevel || compressionLevel > MaximumCompressionLevel)
        {
            throw new ArgumentOutOfRangeException(nameof(compressionLevel),
                $"Compression level must be between {MinimumCompressionLevel} and {MaximumCompressionLevel} for compressor '{Name}' from '{Provider}'.");
        }
    }

    #endregion

    #region Methods

    /// <summary>
    /// Compresses the <see cref="Mat"/> into a byte array.
    /// </summary>
    /// <param name="src">The source <see cref="Mat"/> to compress.</param>
    /// <param name="compressionLevel">The compression level to use.</param>
    /// <returns>A byte array containing the compressed data.</returns>
    public byte[] Compress(Mat src, int compressionLevel)
    {
        ArgumentNullException.ThrowIfNull(src);
        if (src.IsEmpty) return [];
        ValidateCompressionLevel(compressionLevel);
        return CompressCore(src, compressionLevel);
    }

    /// <summary>
    /// Compresses the <see cref="Mat"/> into a byte array.
    /// </summary>
    /// <param name="src">The source <see cref="Mat"/> to compress.</param>
    /// <param name="compressionLevel">The compression level to use.</param>
    /// <returns>A byte array containing the compressed data.</returns>
    public byte[] Compress(Mat src, CompressionLevel compressionLevel)
    {
        return Compress(src, GetCompressionLevel(compressionLevel));
    }

    /// <summary>
    /// Compresses the <see cref="Mat"/> into a byte array using the default compression level (<see cref="DefaultCompressionLevel"/>).
    /// </summary>
    /// <param name="src">The source <see cref="Mat"/> to compress.</param>
    /// <returns>A byte array containing the compressed data.</returns>
    public byte[] Compress(Mat src)
    {
        return Compress(src, GetCompressionLevel(DefaultCompressionLevel));
    }

    /// <summary>
    /// Performs the actual compression. Only called when <paramref name="src"/> is non-empty.
    /// </summary>
    /// <param name="src">The source <see cref="Mat"/> to compress.</param>
    /// <param name="compressionLevel">The compression level to use.</param>
    /// <returns>A byte array containing the compressed data.</returns>
    protected abstract byte[] CompressCore(Mat src, int compressionLevel);

    /// <summary>
    /// Compresses the <see cref="Mat"/> into a byte array asynchronously.
    /// </summary>
    /// <param name="src">The source <see cref="Mat"/> to compress.</param>
    /// <param name="compressionLevel">The compression level to use.</param>
    /// <param name="cancellationToken">A token to monitor for cancellation requests.</param>
    /// <returns>A task that represents the asynchronous operation. The task result contains a byte array with the compressed data.</returns>
    public async Task<byte[]> CompressAsync(Mat src, int compressionLevel,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(src);
        if (src.IsEmpty) return [];
        ValidateCompressionLevel(compressionLevel);
        return await CompressCoreAsync(src, compressionLevel, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Compresses the <see cref="Mat"/> into a byte array asynchronously.
    /// </summary>
    /// <param name="src">The source <see cref="Mat"/> to compress.</param>
    /// <param name="compressionLevel">The compression level to use.</param>
    /// <param name="cancellationToken">A token to monitor for cancellation requests.</param>
    /// <returns>A task that represents the asynchronous operation. The task result contains a byte array with the compressed data.</returns>
    public async Task<byte[]> CompressAsync(Mat src, CompressionLevel compressionLevel,
        CancellationToken cancellationToken = default)
    {
        return await CompressAsync(src, GetCompressionLevel(compressionLevel), cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Compresses the <see cref="Mat"/> into a byte array asynchronously.
    /// </summary>
    /// <param name="src">The source <see cref="Mat"/> to compress.</param>
    /// <param name="cancellationToken">A token to monitor for cancellation requests.</param>
    /// <returns>A task that represents the asynchronous operation. The task result contains a byte array with the compressed data.</returns>
    public async Task<byte[]> CompressAsync(Mat src, CancellationToken cancellationToken = default)
    {
        return await CompressAsync(src, GetCompressionLevel(DefaultCompressionLevel), cancellationToken)
            .ConfigureAwait(false);
    }

    /// <summary>
    /// Compresses the <see cref="Mat"/> into a byte array asynchronously. This method offloads the compression work to a background thread, allowing the calling thread to continue executing without blocking. It is useful for scenarios where compression may take a significant amount of time and you want to keep the UI responsive or perform other tasks concurrently.
    /// </summary>
    /// <param name="src">The source <see cref="Mat"/> to compress.</param>
    /// <param name="compressLevel">The compression level to use.</param>
    /// <param name="cancellationToken">A token to monitor for cancellation requests.</param>
    /// <returns>A task that represents the asynchronous operation. The task result contains a byte array with the compressed data.</returns>
    protected virtual async Task<byte[]> CompressCoreAsync(Mat src, int compressLevel,
        CancellationToken cancellationToken = default)
    {
        return await Task.Run(() => CompressCore(src, compressLevel), cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Decompresses the <see cref="Mat"/> from a byte array.
    /// </summary>
    /// <param name="compressedBytes">The byte array containing the compressed data.</param>
    /// <param name="dst">The destination <see cref="Mat"/> to store the decompressed data.</param>
    public void Decompress(byte[] compressedBytes, Mat dst)
    {
        ArgumentNullException.ThrowIfNull(compressedBytes);
        ArgumentNullException.ThrowIfNull(dst);
        if (compressedBytes.Length == 0) return;
        DecompressCore(compressedBytes, dst);
    }

    /// <summary>
    /// Decompresses the <see cref="Mat"/> from a byte array asynchronously.
    /// </summary>
    /// <param name="compressedBytes">The byte array containing the compressed data.</param>
    /// <param name="dst">The destination <see cref="Mat"/> to store the decompressed data.</param>
    /// <param name="cancellationToken">A token to monitor for cancellation requests.</param>
    /// <returns>A task that represents the asynchronous operation.</returns>
    public async Task DecompressAsync(byte[] compressedBytes, Mat dst, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(compressedBytes);
        ArgumentNullException.ThrowIfNull(dst);
        await DecompressCoreAsync(compressedBytes, dst, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Performs the actual decompression. Only called when <paramref name="compressedBytes"/> is non-empty.
    /// </summary>
    /// <param name="compressedBytes">The byte array containing the compressed data.</param>
    /// <param name="dst">The destination <see cref="Mat"/> to store the decompressed data.</param>
    protected abstract void DecompressCore(byte[] compressedBytes, Mat dst);

    /// <summary>
    /// Decompresses the <see cref="Mat"/> from a byte array asynchronously. This method offloads the decompression work to a background thread, allowing the calling thread to continue executing without blocking. It is useful for scenarios where decompression may take a significant amount of time and you want to keep the UI responsive or perform other tasks concurrently.
    /// </summary>
    /// <param name="compressedBytes">The byte array containing the compressed data.</param>
    /// <param name="dst">The destination <see cref="Mat"/> to store the decompressed data.</param>
    /// <param name="cancellationToken">A token to monitor for cancellation requests.</param>
    /// <returns>A task that represents the asynchronous operation.</returns>
    protected virtual async Task DecompressCoreAsync(byte[] compressedBytes, Mat dst,
        CancellationToken cancellationToken = default)
    {
        await Task.Run(() => DecompressCore(compressedBytes, dst), cancellationToken).ConfigureAwait(false);
    }

    #endregion

    #region Equality Members

    /// <inheritdoc />
    public bool Equals(MatCompressor? other)
    {
        if (other is null) return false;
        if (ReferenceEquals(this, other)) return true;
        return Id == other.Id;
    }

    /// <inheritdoc />
    public override bool Equals(object? obj)
    {
        if (obj is null) return false;
        if (ReferenceEquals(this, obj)) return true;
        return obj is MatCompressor other && Equals(other);
    }

    /// <inheritdoc />
    public override int GetHashCode()
    {
        return StringComparer.Ordinal.GetHashCode(Id);
    }

    #endregion

    #region Static Methods

    /// <summary>
    /// Gets a compressor by its <see cref="Id"/> from the collection of available compressors.
    /// </summary>
    /// <param name="id">The ID of the compressor.</param>
    /// <returns>The compressor with the specified ID, or null if not found.</returns>
    public static MatCompressor? GetCompressorById(string id)
    {
        if (string.IsNullOrWhiteSpace(id)) return null;
        return AvailableCompressors.FirstOrDefault(compressor =>
            compressor.Id.Equals(id, StringComparison.OrdinalIgnoreCase));
    }

    /// <summary>
    /// Gets a compressor by its <see cref="Name"/> from the collection of available compressors.
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
    /// Creates a writable stream over a sparse compression buffer.
    /// </summary>
    /// <param name="buffer">The sparse byte buffer.</param>
    /// <returns>A writable stream backed by <paramref name="buffer"/>.</returns>
    protected static Stream CreateCompressionStream(SparseBufferWriter<byte> buffer)
    {
        return Stream.Create(buffer, true);
    }

    #endregion
}