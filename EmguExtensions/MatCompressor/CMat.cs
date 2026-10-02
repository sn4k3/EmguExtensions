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

using System.Drawing;
using System.IO.Compression;
using System.IO.Hashing;
using Emgu.CV;
using Emgu.CV.CvEnum;

namespace EmguExtensions;

/// <summary>
/// Represents a compressed <see cref="Mat"/> that can be compressed and decompressed using multiple <see cref="MatCompressor"/>s.<br/>
/// This allows to have a high count of <see cref="CMat"/>s in memory without using too much memory.
/// </summary>
/// <remarks>
/// All members are thread-safe. State is guarded by a reader/writer lock: every public property read takes the read lock,
/// every mutation takes the write lock, and members that read several values at once (ratios, equality, <see cref="ToString"/>, ...)
/// work on a single consistent snapshot.
/// </remarks>
public class CMat : IEquatable<CMat>
{
    #region Members

    private readonly ReaderWriterLockSlim _rwLock = new();
    private ulong? _hash;
    private byte[] _compressedBytes = [];
    private bool _isInitialized;
    private bool _isCompressed;
    private int _thresholdToCompress = 512;
    private int _width;
    private int _height;
    private DepthType _depth = DepthType.Cv8U;
    private int _channels = 1;
    private Rectangle _roi;
    private CompressionLevel _compressionLevel = MatCompressor.DefaultCompressionLevel;
    private MatCompressor _compressor = MatCompressor.DefaultCompressor;
    private MatCompressor _decompressor = MatCompressor.DefaultCompressor;

    /// <summary>
    /// A consistent, immutable view of the state, captured under a single read lock.
    /// </summary>
    private readonly record struct State(
        byte[] Bytes,
        ulong? Hash,
        bool IsInitialized,
        bool IsCompressed,
        int Width,
        int Height,
        DepthType Depth,
        int Channels,
        Rectangle Roi,
        MatCompressor Decompressor,
        CompressionLevel CompressionLevel
    )
    {
        public int ElementSize => Depth.ByteCount * Channels;

        public int UncompressedLength =>
            (Roi.Width <= 0 || Roi.Height <= 0 ? Width * Height : Roi.Width * Roi.Height)
            * ElementSize;
    }

    #endregion

    #region Properties

    /// <summary>
    /// Gets the compressed bytes that have been compressed with <see cref="Decompressor"/>.
    /// </summary>
    /// <remarks>
    /// The returned array is the internal buffer, not a copy. Treat it as read-only: mutating it breaks the cached <see cref="Hash"/>
    /// and the thread-safety guarantees of this class.
    /// </remarks>
    public byte[] CompressedBytes => Read(ref _compressedBytes);

    /// <summary>
    /// Gets the XxHash3 hash of the <see cref="CompressedBytes"/>.
    /// </summary>
    public ulong Hash => ResolveHash(GetState());

    /// <summary>
    /// Gets a value indicating whether the <see cref="CompressedBytes"/> have ever been set.
    /// </summary>
    public bool IsInitialized => Read(ref _isInitialized);

    /// <summary>
    /// Gets a value indicating whether the <see cref="CompressedBytes"/> are compressed or raw bytes.
    /// </summary>
    public bool IsCompressed => Read(ref _isCompressed);

    /// <summary>
    /// Gets or sets the threshold in bytes to compress the data. Mat's equal to or less than this size will not be compressed.
    /// </summary>
    public int ThresholdToCompress
    {
        get => Read(ref _thresholdToCompress);
        set => Write(ref _thresholdToCompress, value);
    }

    /// <summary>
    /// Gets the cached width of the <see cref="Mat"/> that was compressed.
    /// </summary>
    public int Width => Read(ref _width);

    /// <summary>
    /// Gets the cached height of the <see cref="Mat"/> that was compressed.
    /// </summary>
    public int Height => Read(ref _height);

    /// <summary>
    /// Gets the cached size of the <see cref="Mat"/> that was compressed.
    /// </summary>
    public Size Size
    {
        get
        {
            var state = GetState();
            return new Size(state.Width, state.Height);
        }
    }

    /// <summary>
    /// Gets the cached depth of the <see cref="Mat"/> that was compressed.
    /// </summary>
    public DepthType Depth => Read(ref _depth);

    /// <summary>
    /// Gets the cached number of channels of the <see cref="Mat"/> that was compressed.
    /// </summary>
    public int Channels => Read(ref _channels);

    /// <summary>
    /// Gets the size, in bytes, of a single element in the data structure, calculated as the product of the depth's
    /// byte count and the number of channels.
    /// </summary>
    public int ElementSize => GetState().ElementSize;

    /// <summary>
    /// Gets the cached ROI of the <see cref="Mat"/> that was compressed.
    /// </summary>
    public Rectangle Roi => Read(ref _roi);

    /// <summary>
    /// Gets or sets the <see cref="CompressionLevel"/> that will be used to compress the <see cref="Mat"/> if the <see cref="Compressor"/> supports it. Default is <see cref="MatCompressor.DefaultCompressionLevel"/> contingencies.
    /// </summary>
    public CompressionLevel CompressionLevel
    {
        get => Read(ref _compressionLevel);
        set => Write(ref _compressionLevel, value);
    }

    /// <summary>
    /// Gets or sets the <see cref="MatCompressor"/> that will be used to compress and decompress the <see cref="Mat"/> contingencies.
    /// </summary>
    public MatCompressor Compressor
    {
        get => Read(ref _compressor);
        set
        {
            ArgumentNullException.ThrowIfNull(value);
            Write(ref _compressor, value);
        }
    }

    /// <summary>
    /// Gets the <see cref="MatCompressor"/> that will be used to decompress the <see cref="Mat"/>.
    /// </summary>
    public MatCompressor Decompressor => Read(ref _decompressor);

    /// <summary>
    /// Gets a value indicating whether the <see cref="CompressedBytes"/> are empty.
    /// </summary>
    public bool IsEmpty => CompressedLength == 0;

    /// <summary>
    /// Gets the length of the <see cref="CompressedBytes"/>.
    /// </summary>
    public int CompressedLength => Read(ref _compressedBytes).Length;

    /// <summary>
    /// Gets the uncompressed length of the <see cref="Mat"/> in bytes, aka bitmap size.
    /// </summary>
    public int UncompressedLength => GetState().UncompressedLength;

    /// <summary>
    /// Gets the compression ratio of the <see cref="CompressedBytes"/> to the <see cref="UncompressedLength"/>.
    /// </summary>
    public float CompressionRatio
    {
        get
        {
            var state = GetState();
            return ComputeCompressionRatio(state.UncompressedLength, state.Bytes.Length);
        }
    }

    /// <summary>
    /// Gets the compression percentage of the <see cref="CompressedBytes"/> to the <see cref="UncompressedLength"/>.
    /// </summary>
    public float CompressionPercentage
    {
        get
        {
            var state = GetState();
            return ComputeCompressionPercentage(state.UncompressedLength, state.Bytes.Length);
        }
    }

    /// <summary>
    /// Gets the compression efficiency percentage of the <see cref="CompressedBytes"/> to the <see cref="UncompressedLength"/>.
    /// </summary>
    public float CompressionEfficiency
    {
        get
        {
            var state = GetState();
            var uncompressedLength = state.UncompressedLength;
            var compressedLength = state.Bytes.Length;
            if (uncompressedLength == 0 || compressedLength == 0)
                return 0;
            return MathF.Round(
                uncompressedLength * 100f / compressedLength,
                2,
                MidpointRounding.AwayFromZero
            );
        }
    }

    /// <summary>
    /// Gets the number of bytes saved by compressing the <see cref="Mat"/>.
    /// </summary>
    public int SavedBytes
    {
        get
        {
            var state = GetState();
            return state.UncompressedLength - state.Bytes.Length;
        }
    }

    /// <summary>
    /// Gets or sets the <see cref="Mat"/> that will be compressed and decompressed.<br/>
    /// Every time the <see cref="Mat"/> is accessed, it will be de/compressed.
    /// </summary>
    public Mat Mat
    {
        get => Decompress();
        set => Compress(value);
    }

    /// <summary>
    /// Gets the <see cref="Mat"/> asynchronously by decompressing <see cref="CompressedBytes"/> on a background thread.<br/>
    /// Every time this property is accessed a new decompression task is started.
    /// </summary>
    public Task<Mat> MatAsync => DecompressAsync();

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the <see cref="CMat"/> class with the specified dimensions, depth, and channel count.
    /// </summary>
    /// <param name="width">The width of the Mat.</param>
    /// <param name="height">The height of the Mat.</param>
    /// <param name="depth">The depth type of the Mat.</param>
    /// <param name="channels">The number of channels.</param>
    public CMat(
        int width = 0,
        int height = 0,
        DepthType depth = DepthType.Cv8U,
        int channels = 1
    )
    {
        _width = width;
        _height = height;
        _depth = depth;
        _channels = channels;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CMat"/> class with the specified size, depth, and channel count.
    /// </summary>
    /// <param name="size">The size of the Mat.</param>
    /// <param name="depth">The depth type of the Mat.</param>
    /// <param name="channels">The number of channels.</param>
    public CMat(Size size, DepthType depth = DepthType.Cv8U, int channels = 1)
        : this(size.Width, size.Height, depth, channels) { }

    /// <summary>
    /// Initializes a new instance of the <see cref="CMat"/> class with the specified compressor, dimensions, depth, and channel count.
    /// </summary>
    /// <param name="compressor">The compressor to use for compression and decompression.</param>
    /// <param name="width">The width of the Mat.</param>
    /// <param name="height">The height of the Mat.</param>
    /// <param name="depth">The depth type of the Mat.</param>
    /// <param name="channels">The number of channels.</param>
    public CMat(
        MatCompressor compressor,
        int width = 0,
        int height = 0,
        DepthType depth = DepthType.Cv8U,
        int channels = 1
    )
        : this(width, height, depth, channels)
    {
        ArgumentNullException.ThrowIfNull(compressor);
        _compressor = compressor;
        _decompressor = compressor;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CMat"/> class with the specified compressor, size, depth, and channel count.
    /// </summary>
    /// <param name="compressor">The compressor to use for compression and decompression.</param>
    /// <param name="size">The size of the Mat.</param>
    /// <param name="depth">The depth type of the Mat.</param>
    /// <param name="channels">The number of channels.</param>
    public CMat(
        MatCompressor compressor,
        Size size,
        DepthType depth = DepthType.Cv8U,
        int channels = 1
    )
        : this(compressor, size.Width, size.Height, depth, channels) { }

    /// <summary>
    /// Initializes a new instance of the <see cref="CMat"/> class by compressing the specified <see cref="Mat"/>.
    /// </summary>
    /// <param name="mat">The Mat to compress.</param>
    /// <param name="compressor">The compressor to use, or <see langword="null"/> to use <see cref="MatCompressor.DefaultCompressor"/>.</param>
    /// <param name="compressionLevel">The compression level to use, or <see langword="null"/> to use <see cref="MatCompressor.DefaultCompressionLevel"/>.</param>
    /// <remarks>To create an async CMat, prefer empty constructor and use CompressAsync method.</remarks>
    public CMat(
        Mat mat,
        MatCompressor? compressor = null,
        CompressionLevel? compressionLevel = null
    )
    {
        ArgumentNullException.ThrowIfNull(mat);
        if (compressor is not null)
        {
            _compressor = compressor;
            _decompressor = compressor;
        }

        _compressionLevel = compressionLevel ?? MatCompressor.DefaultCompressionLevel;

        Compress(mat);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CMat"/> class by compressing the specified <see cref="MatRoi"/>.
    /// </summary>
    /// <param name="matRoi">The MatRoi to compress.</param>
    /// <param name="compressor">The compressor to use, or <see langword="null"/> to use <see cref="MatCompressor.DefaultCompressor"/>.</param>
    /// <param name="compressionLevel">The compression level to use, or <see langword="null"/> to use <see cref="MatCompressor.DefaultCompressionLevel"/>.</param>
    /// <remarks>To create an async CMat, prefer empty constructor and use CompressAsync method.</remarks>
    public CMat(
        MatRoi matRoi,
        MatCompressor? compressor = null,
        CompressionLevel? compressionLevel = null
    )
    {
        ArgumentNullException.ThrowIfNull(matRoi);
        if (compressor is not null)
        {
            _compressor = compressor;
            _decompressor = compressor;
        }

        _compressionLevel = compressionLevel ?? MatCompressor.DefaultCompressionLevel;

        Compress(matRoi);
    }

    #endregion

    #region Locking helpers

    private static float ComputeCompressionRatio(int uncompressedLength, int compressedLength)
    {
        if (
            uncompressedLength == 0
            || compressedLength == 0
            || compressedLength == uncompressedLength
        )
            return 0;
        return MathF.Round(
            (float)uncompressedLength / compressedLength,
            2,
            MidpointRounding.AwayFromZero
        );
    }

    private static float ComputeCompressionPercentage(int uncompressedLength, int compressedLength)
    {
        if (
            compressedLength == 0
            || uncompressedLength == 0
            || compressedLength == uncompressedLength
        )
            return 0;
        return MathF.Round(
            100 - (compressedLength * 100f / uncompressedLength),
            2,
            MidpointRounding.AwayFromZero
        );
    }

    /// <summary>
    /// Reads a field under the read lock. Must not be called while the current thread already holds a lock.
    /// </summary>
    private T Read<T>(ref T field)
    {
        _rwLock.EnterReadLock();
        try
        {
            return field;
        }
        finally
        {
            _rwLock.ExitReadLock();
        }
    }

    /// <summary>
    /// Writes a field under the write lock. Must not be called while the current thread already holds a lock.
    /// </summary>
    private void Write<T>(ref T field, T value)
    {
        _rwLock.EnterWriteLock();
        try
        {
            field = value;
        }
        finally
        {
            _rwLock.ExitWriteLock();
        }
    }

    /// <summary>
    /// Captures a consistent snapshot of the state under a single read lock.
    /// </summary>
    private State GetState()
    {
        _rwLock.EnterReadLock();
        try
        {
            return new State(
                _compressedBytes,
                _hash,
                _isInitialized,
                _isCompressed,
                _width,
                _height,
                _depth,
                _channels,
                _roi,
                _decompressor,
                _compressionLevel
            );
        }
        finally
        {
            _rwLock.ExitReadLock();
        }
    }

    /// <summary>
    /// Gets the hash of a snapshot, computing it outside of any lock when it is not cached yet, and caches it
    /// when the bytes have not been replaced in the meantime.
    /// </summary>
    private ulong ResolveHash(in State state)
    {
        if (state.Hash.HasValue)
            return state.Hash.Value;

        // The compressed array is never mutated in place, only replaced, so it is safe to hash without holding a lock
        var hash = XxHash3.HashToUInt64(state.Bytes);

        _rwLock.EnterWriteLock();
        try
        {
            if (ReferenceEquals(_compressedBytes, state.Bytes))
                _hash = hash;
        }
        finally
        {
            _rwLock.ExitWriteLock();
        }

        return hash;
    }

    /// <summary>
    /// Sets the compressed bytes and the state derived from them. Caller must hold the write lock.
    /// </summary>
    private void SetBytesInternal(byte[] value)
    {
        _compressedBytes = value;
        _hash = null;
        _isInitialized = true;
        _isCompressed = value.Length != 0;
    }

    /// <summary>
    /// Clears the compressed bytes and the ROI when not empty. Caller must hold the write lock.
    /// </summary>
    private void ClearBytesInternal()
    {
        if (_compressedBytes.Length == 0)
            return;
        SetBytesInternal([]);
        _roi = Rectangle.Empty;
    }

    #endregion

    #region Compress/Decompress

    /// <summary>
    /// Changes the <see cref="Compressor"/> and optionally re-encodes the <see cref="Mat"/> with the new <paramref name="compressor"/> if the <see cref="Decompressor"/> is different from the set <paramref name="compressor"/>.
    /// </summary>
    /// <param name="compressor">New compressor</param>
    /// <param name="compressionLevel">The compression level to use.</param>
    /// <param name="reEncodeWithNewCompressor">True to re-encodes the <see cref="Mat"/> with the new <see cref="Compressor"/>, otherwise false.</param>
    /// <returns>True if compressor has been changed, otherwise false.</returns>
    public bool ChangeCompressor(
        MatCompressor compressor,
        CompressionLevel compressionLevel,
        bool reEncodeWithNewCompressor = false
    )
    {
        ArgumentNullException.ThrowIfNull(compressor);
        _rwLock.EnterWriteLock();
        try
        {
            var willReEncode =
                reEncodeWithNewCompressor
                && _compressedBytes.Length != 0
                && (!_decompressor.Equals(compressor) || _compressionLevel != compressionLevel);
            if (
                _compressor.Equals(compressor)
                && _compressionLevel == compressionLevel
                && !willReEncode
            )
                return false; // Nothing to change

            _compressor = compressor;
            _compressionLevel = compressionLevel;

            if (willReEncode)
            {
                var lastWidth = _width;
                var lastHeight = _height;
                var lastRoi = _roi;
                using var mat = RawDecompressInternal();
                try
                {
                    CompressInternal(mat);
                }
                finally
                {
                    _width = lastWidth;
                    _height = lastHeight;
                    _roi = lastRoi;
                }
            }

            return true;
        }
        finally
        {
            _rwLock.ExitWriteLock();
        }
    }

    /// <summary>
    /// Changes the <see cref="Compressor"/> and optionally re-encodes the <see cref="Mat"/> with the new <paramref name="compressor"/> if the <see cref="Decompressor"/> is different from the set <paramref name="compressor"/>.
    /// </summary>
    /// <param name="compressor">New compressor</param>
    /// <param name="reEncodeWithNewCompressor">True to re-encodes the <see cref="Mat"/> with the new <see cref="Compressor"/>, otherwise false.</param>
    /// <returns>True if compressor has been changed, otherwise false.</returns>
    public bool ChangeCompressor(MatCompressor compressor, bool reEncodeWithNewCompressor = false)
    {
        return ChangeCompressor(compressor, CompressionLevel, reEncodeWithNewCompressor);
    }

    /// <summary>
    /// Changes the <see cref="Compressor"/> and optionally re-encodes the <see cref="Mat"/> with the new <paramref name="compressor"/> if the <see cref="Decompressor"/> is different from the set <paramref name="compressor"/>.
    /// </summary>
    /// <param name="compressor">New compressor</param>
    /// <param name="compressionLevel">The compression level to use.</param>
    /// <param name="reEncodeWithNewCompressor">True to re-encodes the <see cref="Mat"/> with the new <see cref="Compressor"/>, otherwise false.</param>
    /// <param name="cancellationToken">A token to cancel the operation.</param>
    /// <returns>True if compressor has been changed, otherwise false.</returns>
    public Task<bool> ChangeCompressorAsync(
        MatCompressor compressor,
        CompressionLevel compressionLevel,
        bool reEncodeWithNewCompressor = false,
        CancellationToken cancellationToken = default
    )
    {
        return Task.Run(
            () => ChangeCompressor(compressor, compressionLevel, reEncodeWithNewCompressor),
            cancellationToken
        );
    }

    /// <summary>
    /// Changes the <see cref="Compressor"/> and optionally re-encodes the <see cref="Mat"/> with the new <paramref name="compressor"/> if the <see cref="Decompressor"/> is different from the set <paramref name="compressor"/>.
    /// </summary>
    /// <param name="compressor">New compressor</param>
    /// <param name="reEncodeWithNewCompressor">True to re-encodes the <see cref="Mat"/> with the new <see cref="Compressor"/>, otherwise false.</param>
    /// <param name="cancellationToken">A token to cancel the operation.</param>
    /// <returns>True if compressor has been changed, otherwise false.</returns>
    public Task<bool> ChangeCompressorAsync(
        MatCompressor compressor,
        bool reEncodeWithNewCompressor = false,
        CancellationToken cancellationToken = default
    )
    {
        return ChangeCompressorAsync(
            compressor,
            CompressionLevel,
            reEncodeWithNewCompressor,
            cancellationToken
        );
    }

    /// <summary>
    /// Sets the <see cref="CompressedBytes"/> to an empty byte array and sets <see cref="IsCompressed"/> to false.
    /// </summary>
    /// <remarks>
    /// This is a no-op when the <see cref="CompressedBytes"/> are already empty, in which case <see cref="IsInitialized"/> is left untouched.
    /// Use <see cref="SetEmptyCompressedBytes(bool)"/> to set <see cref="IsInitialized"/> explicitly.
    /// </remarks>
    public void SetEmptyCompressedBytes()
    {
        _rwLock.EnterWriteLock();
        try
        {
            ClearBytesInternal();
        }
        finally
        {
            _rwLock.ExitWriteLock();
        }
    }

    /// <summary>
    /// Sets the <see cref="CompressedBytes"/> to an empty byte array and sets <see cref="IsCompressed"/> to false.
    /// </summary>
    /// <param name="isInitialized">Sets the <see cref="IsInitialized"/> to a known state.</param>
    public void SetEmptyCompressedBytes(bool isInitialized)
    {
        _rwLock.EnterWriteLock();
        try
        {
            ClearBytesInternal();
            _isInitialized = isInitialized;
        }
        finally
        {
            _rwLock.ExitWriteLock();
        }
    }

    /// <summary>
    /// Sets the <see cref="CompressedBytes"/> to an empty byte array, sets <see cref="IsCompressed"/> to false and extract size, depth and channels from a <see cref="Mat"/>.
    /// </summary>
    /// <param name="src">Source Mat to extract Size, Depth and Channels</param>
    public void SetEmptyCompressedBytes(Mat src)
    {
        ArgumentNullException.ThrowIfNull(src);
        _rwLock.EnterWriteLock();
        try
        {
            ClearBytesInternal();
            _width = src.Width;
            _height = src.Height;
            _depth = src.Depth;
            _channels = src.NumberOfChannels;
        }
        finally
        {
            _rwLock.ExitWriteLock();
        }
    }

    /// <summary>
    /// Sets the <see cref="CompressedBytes"/> to an empty byte array, sets <see cref="IsCompressed"/> to false and extract size, depth and channels from a <see cref="Mat"/>.
    /// </summary>
    /// <param name="src">Source Mat to extract Size, Depth and Channels</param>
    /// <param name="isInitialized">Sets the <see cref="IsInitialized"/> to a known state.</param>
    public void SetEmptyCompressedBytes(Mat src, bool isInitialized)
    {
        ArgumentNullException.ThrowIfNull(src);
        _rwLock.EnterWriteLock();
        try
        {
            ClearBytesInternal();
            _width = src.Width;
            _height = src.Height;
            _depth = src.Depth;
            _channels = src.NumberOfChannels;
            _isInitialized = isInitialized;
        }
        finally
        {
            _rwLock.ExitWriteLock();
        }
    }

    /// <summary>
    /// Sets the <see cref="CompressedBytes"/> and <see cref="Compressor"/> and <see cref="Decompressor"/>.
    /// </summary>
    /// <param name="compressedBytes">The compressed byte array to set. The <see cref="CMat"/> takes ownership of the array (it is not copied): do not modify it afterwards.</param>
    /// <param name="decompressor">The decompressor that matches the compressed data.</param>
    /// <param name="setCompressor">If <see langword="true"/>, also sets the <see cref="Compressor"/> to the specified decompressor.</param>
    /// <remarks>
    /// The <see cref="Roi"/> is cleared, the bytes are expected to describe the full <see cref="Width"/> x <see cref="Height"/> area.
    /// </remarks>
    public void SetCompressedBytes(
        byte[] compressedBytes,
        MatCompressor decompressor,
        bool setCompressor = true
    )
    {
        ArgumentNullException.ThrowIfNull(compressedBytes);
        ArgumentNullException.ThrowIfNull(decompressor);
        _rwLock.EnterWriteLock();
        try
        {
            SetBytesInternal(compressedBytes);
            _roi = Rectangle.Empty;
            if (setCompressor)
                _compressor = decompressor;
            _decompressor = decompressor;
            if (ReferenceEquals(decompressor, MatCompressorNone.Instance))
                _isCompressed = false;
        }
        finally
        {
            _rwLock.ExitWriteLock();
        }
    }

    /// <summary>
    /// Sets the <see cref="CompressedBytes"/> to uncompressed bitmap data. Caller must hold the write lock.
    /// </summary>
    /// <param name="src">The source Mat whose raw data will be stored uncompressed.</param>
    private void SetUncompressed(Mat src)
    {
        SetBytesInternal(src.ToArray());
        _isCompressed = false;
        _decompressor = MatCompressorNone.Instance;
    }

    /// <summary>
    /// Compresses the <see cref="Mat"/> into a byte array.
    /// </summary>
    /// <param name="src">The Mat to compress.</param>
    public void Compress(Mat src)
    {
        ArgumentNullException.ThrowIfNull(src);
        _rwLock.EnterWriteLock();
        try
        {
            CompressInternal(src);
        }
        finally
        {
            _rwLock.ExitWriteLock();
        }
    }

    /// <summary>
    /// Internal compress implementation without locking. Caller must hold the write lock.
    /// </summary>
    private void CompressInternal(Mat src)
    {
        _width = src.Width;
        _height = src.Height;
        _depth = src.Depth;
        _channels = src.NumberOfChannels;
        _roi = Rectangle.Empty;

        if (src.IsEmpty)
        {
            SetBytesInternal([]);
            return;
        }

        var srcLength = src.ByteCountInt32;

        // Do not compress if the size is smaller or equal to the threshold, or if there is nothing to compress with
        if (
            srcLength <= _thresholdToCompress
            || ReferenceEquals(_compressor, MatCompressorNone.Instance)
        )
        {
            SetUncompressed(src);
            return;
        }

        try
        {
            var compressed = _compressor.Compress(src, _compressionLevel);
            if (compressed.Length > 0 && compressed.Length < srcLength) // Compressed ok
            {
                SetBytesInternal(compressed);
                _decompressor = _compressor;
            }
            else // Empty result, or compressed size is larger or equal to uncompressed size, store raw
            {
                SetUncompressed(src);
            }
        }
        // Cannot compress due some codec error (e.g. unsupported type), store raw instead.
        // Argument errors (e.g. invalid level) are programming errors and must surface.
        catch (Exception ex)
            when (ex
                    is not (
                        OutOfMemoryException
                        or AccessViolationException
                        or OperationCanceledException
                        or ArgumentException
                    )
            )
        {
            SetUncompressed(src);
        }
    }

    /// <summary>
    /// Compresses the <see cref="MatRoi"/> into a byte array.
    /// </summary>
    /// <param name="src">The MatRoi to compress.</param>
    public void Compress(MatRoi src)
    {
        ArgumentNullException.ThrowIfNull(src);
        _rwLock.EnterWriteLock();
        try
        {
            if (src.Roi.Width <= 0 || src.Roi.Height <= 0)
            {
                _width = src.SourceMat.Width;
                _height = src.SourceMat.Height;
                _depth = src.SourceMat.Depth;
                _channels = src.SourceMat.NumberOfChannels;
                _roi = Rectangle.Empty;
                SetBytesInternal([]);
                return;
            }

            if (src.IsSourceSameSizeOfRoi)
            {
                CompressInternal(src.SourceMat);
            }
            else
            {
                CompressInternal(src.RoiMat);
                _width = src.SourceMat.Width;
                _height = src.SourceMat.Height;
                _roi = src.Roi;
            }
        }
        finally
        {
            _rwLock.ExitWriteLock();
        }
    }

    /// <summary>
    /// Compresses the <see cref="Mat"/> into a byte array asynchronously.
    /// </summary>
    /// <param name="src">The Mat to compress.</param>
    /// <param name="cancellationToken">A token to cancel the operation.</param>
    /// <returns>A task representing the asynchronous compression operation.</returns>
    public Task CompressAsync(Mat src, CancellationToken cancellationToken = default)
    {
        return Task.Run(() => Compress(src), cancellationToken);
    }

    /// <summary>
    /// Compresses the <see cref="MatRoi"/> into a byte array asynchronously.
    /// </summary>
    /// <param name="src">The MatRoi to compress.</param>
    /// <param name="cancellationToken">A token to cancel the operation.</param>
    /// <returns>A task representing the asynchronous compression operation.</returns>
    public Task CompressAsync(MatRoi src, CancellationToken cancellationToken = default)
    {
        return Task.Run(() => Compress(src), cancellationToken);
    }

    /// <summary>
    /// Decompresses the <see cref="CompressedBytes"/> into a new <see cref="Mat"/> without expanding into the original <see cref="Mat"/> if there is a <see cref="Roi"/>.
    /// </summary>
    /// <returns>Returns a <see cref="Mat"/> with size of <see cref="Roi"/> if is not empty, otherwise returns the original <see cref="Size"/></returns>
    public Mat RawDecompress()
    {
        _rwLock.EnterReadLock();
        try
        {
            return RawDecompressInternal();
        }
        finally
        {
            _rwLock.ExitReadLock();
        }
    }

    /// <summary>
    /// Internal raw decompress implementation without locking. Caller must hold at least a read lock.
    /// </summary>
    private Mat RawDecompressInternal()
    {
        var hasRoi = _roi.Width > 0 && _roi.Height > 0;

        if (_compressedBytes.Length == 0)
            return hasRoi
                ? EmguCvExtensions.InitMat(_roi.Size, _channels, _depth)
                : CreateMatZerosInternal();

        var mat = hasRoi ? new Mat(_roi.Size, _depth, _channels) : CreateMatInternal();
        try
        {
            if (_isCompressed)
            {
                _decompressor.Decompress(_compressedBytes, mat);
            }
            else
            {
                MatCompressorNone.Instance.Decompress(_compressedBytes, mat);
            }

            ValidateDecompressed(mat, hasRoi ? _roi.Size : new Size(_width, _height));
            return mat;
        }
        catch
        {
            mat.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Validates that the decompressed <paramref name="mat"/> matches the cached description, when there is one.
    /// </summary>
    private void ValidateDecompressed(Mat mat, Size expectedSize)
    {
        if (expectedSize.Width <= 0 || expectedSize.Height <= 0)
            return; // The size is unknown (e.g. self-describing data set without a size)

        if (
            mat.Size != expectedSize
            || mat.Depth != _depth
            || mat.NumberOfChannels != _channels
        )
        {
            throw new InvalidDataException(
                $"The decompressed data ({mat.Width}x{mat.Height}, {mat.Depth}, {mat.NumberOfChannels} channel(s)) does not match the expected description ({expectedSize.Width}x{expectedSize.Height}, {_depth}, {_channels} channel(s))."
            );
        }
    }

    /// <summary>
    /// Decompresses the <see cref="CompressedBytes"/> into a new <see cref="Mat"/> without expanding into the original <see cref="Mat"/> if there is a <see cref="Roi"/>.
    /// </summary>
    /// <param name="cancellationToken"></param>
    /// <returns>Returns a <see cref="Mat"/> with size of <see cref="Roi"/> if is not empty, otherwise returns the original <see cref="Size"/></returns>
    public Task<Mat> RawDecompressAsync(CancellationToken cancellationToken = default)
    {
        return Task.Run(RawDecompress, cancellationToken);
    }

    /// <summary>
    /// Decompresses the <see cref="CompressedBytes"/> into a new <see cref="Mat"/> .
    /// </summary>
    /// <returns></returns>
    public Mat Decompress()
    {
        _rwLock.EnterReadLock();
        try
        {
            if (_compressedBytes.Length == 0)
                return CreateMatZerosInternal();

            var mat = RawDecompressInternal();
            if (_roi.Width <= 0 || _roi.Height <= 0)
                return mat;

            var fullMat = CreateMatZerosInternal();
            try
            {
                using var roi = new Mat(fullMat, _roi);
                mat.CopyTo(roi);
                return fullMat;
            }
            catch
            {
                fullMat.Dispose();
                throw;
            }
            finally
            {
                mat.Dispose();
            }
        }
        finally
        {
            _rwLock.ExitReadLock();
        }
    }

    /// <summary>
    /// Decompresses the <see cref="CompressedBytes"/> into a new <see cref="Mat"/>.
    /// </summary>
    /// <param name="cancellationToken"></param>
    /// <returns></returns>
    public Task<Mat> DecompressAsync(CancellationToken cancellationToken = default)
    {
        return Task.Run(Decompress, cancellationToken);
    }

    #endregion

    #region Utilities

    /// <summary>
    /// Creates a new <see cref="Mat"/> with the same size, depth, and channels as the <see cref="CMat"/>.
    /// </summary>
    /// <returns></returns>
    public Mat CreateMat()
    {
        _rwLock.EnterReadLock();
        try
        {
            return CreateMatInternal();
        }
        finally
        {
            _rwLock.ExitReadLock();
        }
    }

    private Mat CreateMatInternal()
    {
        if (_width <= 0 || _height <= 0)
            return new Mat();
        return new Mat(new Size(_width, _height), _depth, _channels);
    }

    /// <summary>
    /// Create a new <see cref="Mat"/> with the same size, depth, and channels as the <see cref="CMat"/> but with all bytes set to 0.
    /// </summary>
    /// <returns></returns>
    public Mat CreateMatZeros()
    {
        _rwLock.EnterReadLock();
        try
        {
            return CreateMatZerosInternal();
        }
        finally
        {
            _rwLock.ExitReadLock();
        }
    }

    private Mat CreateMatZerosInternal()
    {
        return EmguCvExtensions.InitMat(new Size(_width, _height), _channels, _depth);
    }

    #endregion

    #region Copy and Clone
    /// <summary>
    /// Copies the <see cref="CMat"/> to the <paramref name="dst"/>.
    /// </summary>
    /// <param name="dst"></param>
    public void CopyTo(CMat dst)
    {
        ArgumentNullException.ThrowIfNull(dst);
        if (ReferenceEquals(this, dst))
            return;

        byte[] compressedBytes;
        State state;
        int thresholdToCompress;
        MatCompressor compressor;

        _rwLock.EnterReadLock();
        try
        {
            state = new State(
                _compressedBytes,
                _hash,
                _isInitialized,
                _isCompressed,
                _width,
                _height,
                _depth,
                _channels,
                _roi,
                _decompressor,
                _compressionLevel
            );
            compressedBytes = _compressedBytes.ToArrayPerf(); // Deep copy, the clone must not share the buffer
            thresholdToCompress = _thresholdToCompress;
            compressor = _compressor;
        }
        finally
        {
            _rwLock.ExitReadLock();
        }

        dst._rwLock.EnterWriteLock();
        try
        {
            dst._compressedBytes = compressedBytes;
            dst._hash = state.Hash;
            dst._isInitialized = state.IsInitialized;
            dst._isCompressed = state.IsCompressed;
            dst._thresholdToCompress = thresholdToCompress;
            dst._compressionLevel = state.CompressionLevel;
            dst._compressor = compressor;
            dst._decompressor = state.Decompressor;
            dst._width = state.Width;
            dst._height = state.Height;
            dst._depth = state.Depth;
            dst._channels = state.Channels;
            dst._roi = state.Roi;
        }
        finally
        {
            dst._rwLock.ExitWriteLock();
        }
    }

    /// <summary>
    /// Creates a clone of the <see cref="CMat"/> with the same <see cref="CompressedBytes"/>.
    /// </summary>
    /// <returns></returns>
    public CMat Clone()
    {
        var clone = new CMat();
        CopyTo(clone);
        return clone;
    }
    #endregion

    #region Formatters

    /// <inheritdoc />
    public override string ToString()
    {
        var state = GetState();
        var size = new Size(state.Width, state.Height);
        var uncompressedLength = state.UncompressedLength;
        var compressedLength = state.Bytes.Length;
        var ratio = ComputeCompressionRatio(uncompressedLength, compressedLength);
        var percentage = ComputeCompressionPercentage(uncompressedLength, compressedLength);
        return $"{nameof(Decompressor)}: {state.Decompressor} @ {state.CompressionLevel}, {nameof(Size)}: {size}, {nameof(UncompressedLength)}: {uncompressedLength}, {nameof(CompressedLength)}: {compressedLength}, {nameof(IsCompressed)}: {state.IsCompressed}, {nameof(CompressionRatio)}: {ratio}x, {nameof(CompressionPercentage)}: {percentage}%";
    }

    #endregion

    #region Equality

    /// <inheritdoc />
    public bool Equals(CMat? other)
    {
        if (ReferenceEquals(null, other))
            return false;
        if (ReferenceEquals(this, other))
            return true;

        // Each side is captured under its own lock, they are never nested so no lock ordering issues can happen
        var left = GetState();
        var right = other.GetState();

        return left.IsInitialized == right.IsInitialized
            && left.IsCompressed == right.IsCompressed
            && left.Width == right.Width
            && left.Height == right.Height
            && left.Depth == right.Depth
            && left.Channels == right.Channels
            && left.Roi.Equals(right.Roi)
            && left.Bytes.Length == right.Bytes.Length
            && (
                ReferenceEquals(left.Bytes, right.Bytes)
                || ResolveHash(left) == other.ResolveHash(right)
            );
    }

    /// <inheritdoc />
    public override bool Equals(object? obj)
    {
        if (ReferenceEquals(null, obj))
            return false;
        if (ReferenceEquals(this, obj))
            return true;
        if (obj.GetType() != this.GetType())
            return false;
        return Equals((CMat)obj);
    }

    /// <summary>
    /// Determines whether two <see cref="CMat"/> instances are equal.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns><see langword="true"/> if both instances are equal; otherwise <see langword="false"/>.</returns>
    public static bool operator ==(CMat? left, CMat? right)
    {
        return Equals(left, right);
    }

    /// <summary>
    /// Determines whether two <see cref="CMat"/> instances are not equal.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns><see langword="true"/> if the instances are not equal; otherwise <see langword="false"/>.</returns>
    public static bool operator !=(CMat? left, CMat? right)
    {
        return !Equals(left, right);
    }

    /// <summary>
    /// Gets a hash code based on the content hash (cached after the first call) and the description of the <see cref="Mat"/>.
    /// </summary>
    /// <remarks>As <see cref="CMat"/> is mutable, the hash code changes when the content changes: do not mutate an instance used as a dictionary key.</remarks>
    public override int GetHashCode()
    {
        var state = GetState();
        return HashCode.Combine(
            ResolveHash(state),
            state.Width,
            state.Height,
            state.Depth,
            state.Channels,
            state.Roi
        );
    }

    #endregion
}
