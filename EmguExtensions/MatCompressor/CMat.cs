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

using Emgu.CV;
using Emgu.CV.CvEnum;
using System.Drawing;
using System.IO.Compression;
using System.IO.Hashing;

namespace EmguExtensions;

/// <summary>
/// Represents a compressed <see cref="Mat"/> that can be compressed and decompressed using multiple <see cref="MatCompressor"/>s.<br/>
/// This allows to have a high count of <see cref="CMat"/>s in memory without using too much memory.
/// </summary>
public class CMat : IEquatable<CMat>
{
    #region Members

    private readonly ReaderWriterLockSlim _rwLock = new();
    private ulong? _hash;
    private byte[] _compressedBytes = [];

    #endregion

    #region Properties

    /// <summary>
    /// Gets the compressed bytes that have been compressed with <see cref="Decompressor"/>.
    /// </summary>
    public byte[] CompressedBytes
    {
        get => _compressedBytes;
        private set
        {
            _compressedBytes = value;
            _hash = null;
            IsInitialized = true;
            IsCompressed = value.Length != 0;
        }
    }

    /// <summary>
    /// Gets the XxHash3 hash of the <see cref="CompressedBytes"/>.
    /// </summary>
    public ulong Hash
    {
        get
        {
            _rwLock.EnterReadLock();
            try
            {
                if (_hash.HasValue) return _hash.Value;
            }
            finally
            {
                _rwLock.ExitReadLock();
            }

            _rwLock.EnterUpgradeableReadLock();
            try
            {
                if (_hash.HasValue) return _hash.Value;

                var hash = XxHash3.HashToUInt64(_compressedBytes);
                _rwLock.EnterWriteLock();
                try
                {
                    _hash = hash;
                }
                finally
                {
                    _rwLock.ExitWriteLock();
                }

                return hash;
            }
            finally
            {
                _rwLock.ExitUpgradeableReadLock();
            }
        }
    }

    /// <summary>
    /// Gets a value indicating whether the <see cref="CompressedBytes"/> have ever been set.
    /// </summary>
    public bool IsInitialized { get; private set; }

    /// <summary>
    /// Gets a value indicating whether the <see cref="CompressedBytes"/> are compressed or raw bytes.
    /// </summary>
    public bool IsCompressed { get; private set; }

    /// <summary>
    /// Gets or sets the threshold in bytes to compress the data. Mat's equal to or less than this size will not be compressed.
    /// </summary>
    public int ThresholdToCompress { get; set; } = 512;

    /// <summary>
    /// Gets the cached width of the <see cref="Mat"/> that was compressed.
    /// </summary>
    public int Width { get; private set; }

    /// <summary>
    /// Gets the cached height of the <see cref="Mat"/> that was compressed.
    /// </summary>
    public int Height { get; private set; }

    /// <summary>
    /// Gets the cached size of the <see cref="Mat"/> that was compressed.
    /// </summary>
    public Size Size
    {
        get => new(Width, Height);
        private set
        {
            Width = value.Width;
            Height = value.Height;
        }
    }

    /// <summary>
    /// Gets the cached depth of the <see cref="Mat"/> that was compressed.
    /// </summary>
    public DepthType Depth { get; private set; } = DepthType.Cv8U;

    /// <summary>
    /// Gets the cached number of channels of the <see cref="Mat"/> that was compressed.
    /// </summary>
    public int Channels { get; private set; } = 1;

    /// <summary>
    /// Gets the size, in bytes, of a single element in the data structure, calculated as the product of the depth's
    /// byte count and the number of channels.
    /// </summary>
    public int ElementSize => Depth.ByteCount * Channels;

    /// <summary>
    /// Gets the cached ROI of the <see cref="Mat"/> that was compressed.
    /// </summary>
    public Rectangle Roi { get; private set; }

    /// <summary>
    /// Gets or sets the <see cref="CompressionLevel"/> that will be used to compress the <see cref="Mat"/> if the <see cref="Compressor"/> supports it. Default is <see cref="MatCompressor.DefaultCompressionLevel"/>.
    /// </summary>
    public CompressionLevel CompressionLevel { get; set; } = MatCompressor.DefaultCompressionLevel;

    /// <summary>
    /// Gets or sets the <see cref="MatCompressor"/> that will be used to compress and decompress the <see cref="Mat"/>.
    /// </summary>
    public MatCompressor Compressor { get; set; } = MatCompressor.DefaultCompressor;

    /// <summary>
    /// Gets the <see cref="MatCompressor"/> that will be used to decompress the <see cref="Mat"/>.
    /// </summary>
    public MatCompressor Decompressor { get; private set; } = MatCompressor.DefaultCompressor;

    /// <summary>
    /// Gets a value indicating whether the <see cref="CompressedBytes"/> are empty.
    /// </summary>
    public bool IsEmpty => CompressedLength == 0;

    /// <summary>
    /// Gets the length of the <see cref="CompressedBytes"/>.
    /// </summary>
    public int CompressedLength => CompressedBytes.Length;

    /// <summary>
    /// Gets the uncompressed length of the <see cref="Mat"/> in bytes, aka bitmap size.
    /// </summary>
    public int UncompressedLength => (Roi.Size.IsEmpty ? Width * Height : Roi.Width * Roi.Height) * ElementSize;

    /// <summary>
    /// Gets the compression ratio of the <see cref="CompressedBytes"/> to the <see cref="UncompressedLength"/>.
    /// </summary>
    public float CompressionRatio
    {
        get
        {
            var uncompressedLength = UncompressedLength;
            var compressedLength = CompressedLength;
            if (uncompressedLength == 0 || compressedLength == 0 || compressedLength == uncompressedLength) return 0;
            return MathF.Round((float)uncompressedLength / compressedLength, 2, MidpointRounding.AwayFromZero);
        }
    }

    /// <summary>
    /// Gets the compression percentage of the <see cref="CompressedBytes"/> to the <see cref="UncompressedLength"/>.
    /// </summary>
    public float CompressionPercentage
    {
        get
        {
            var uncompressedLength = UncompressedLength;
            var compressedLength = CompressedLength;
            if (compressedLength == 0 || uncompressedLength == 0 || compressedLength == uncompressedLength) return 0;
            return MathF.Round(100 - (compressedLength * 100f / uncompressedLength), 2, MidpointRounding.AwayFromZero);
        }
    }

    /// <summary>
    /// Gets the compression efficiency percentage of the <see cref="CompressedBytes"/> to the <see cref="UncompressedLength"/>.
    /// </summary>
    public float CompressionEfficiency
    {
        get
        {
            var uncompressedLength = UncompressedLength;
            var compressedLength = CompressedLength;
            if (uncompressedLength == 0 || compressedLength == 0) return 0;
            return MathF.Round(uncompressedLength * 100f / compressedLength, 2, MidpointRounding.AwayFromZero);
        }
    }

    /// <summary>
    /// Gets the number of bytes saved by compressing the <see cref="Mat"/>.
    /// </summary>
    public int SavedBytes => UncompressedLength - CompressedLength;

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
    public CMat(int width = 0, int height = 0, DepthType depth = DepthType.Cv8U, int channels = 1)
    {
        Width = width;
        Height = height;
        Depth = depth;
        Channels = channels;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CMat"/> class with the specified size, depth, and channel count.
    /// </summary>
    /// <param name="size">The size of the Mat.</param>
    /// <param name="depth">The depth type of the Mat.</param>
    /// <param name="channels">The number of channels.</param>
    public CMat(Size size, DepthType depth = DepthType.Cv8U, int channels = 1) : this(size.Width, size.Height, depth, channels)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CMat"/> class with the specified compressor, dimensions, depth, and channel count.
    /// </summary>
    /// <param name="compressor">The compressor to use for compression and decompression.</param>
    /// <param name="width">The width of the Mat.</param>
    /// <param name="height">The height of the Mat.</param>
    /// <param name="depth">The depth type of the Mat.</param>
    /// <param name="channels">The number of channels.</param>
    public CMat(MatCompressor compressor, int width = 0, int height = 0, DepthType depth = DepthType.Cv8U, int channels = 1) : this(width, height, depth, channels)
    {
        Compressor = compressor;
        Decompressor = compressor;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CMat"/> class with the specified compressor, size, depth, and channel count.
    /// </summary>
    /// <param name="compressor">The compressor to use for compression and decompression.</param>
    /// <param name="size">The size of the Mat.</param>
    /// <param name="depth">The depth type of the Mat.</param>
    /// <param name="channels">The number of channels.</param>
    public CMat(MatCompressor compressor, Size size, DepthType depth = DepthType.Cv8U, int channels = 1) : this(compressor, size.Width, size.Height, depth, channels)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CMat"/> class by compressing the specified <see cref="Mat"/>.
    /// </summary>
    /// <param name="mat">The Mat to compress.</param>
    /// <param name="compressor">The compressor to use, or <see langword="null"/> to use the default (Brotli).</param>
    /// <param name="compressionLevel">The compression level to use.</param>
    /// <remarks>To create an async CMat, prefer empty constructor and use CompressAsync method.</remarks>
    public CMat(Mat mat, MatCompressor? compressor = null, CompressionLevel compressionLevel = CompressionLevel.Optimal)
    {
        if (compressor is not null)
        {
            Compressor = compressor;
            Decompressor = compressor;
        }

        CompressionLevel = compressionLevel;

        Compress(mat);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CMat"/> class by compressing the specified <see cref="MatRoi"/>.
    /// </summary>
    /// <param name="matRoi">The MatRoi to compress.</param>
    /// <param name="compressor">The compressor to use, or <see langword="null"/> to use the default (Brotli).</param>
    /// <param name="compressionLevel">The compression level to use.</param>
    /// <remarks>To create an async CMat, prefer empty constructor and use CompressAsync method.</remarks>
    public CMat(MatRoi matRoi, MatCompressor? compressor = null, CompressionLevel compressionLevel = CompressionLevel.Optimal)
    {
        if (compressor is not null)
        {
            Compressor = compressor;
            Decompressor = compressor;
        }

        CompressionLevel = compressionLevel;

        Compress(matRoi);
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
    public bool ChangeCompressor(MatCompressor compressor, CompressionLevel compressionLevel, bool reEncodeWithNewCompressor = false)
    {
        _rwLock.EnterWriteLock();
        try
        {
            bool willReEncode = reEncodeWithNewCompressor
                                && !IsEmpty
                                && (!ReferenceEquals(Decompressor, compressor)
                                    || CompressionLevel != compressionLevel);
            if (ReferenceEquals(Compressor, compressor)
                && CompressionLevel == compressionLevel
                && !willReEncode) return false; // Nothing to change
            Compressor = compressor;
            CompressionLevel = compressionLevel;

            if (willReEncode)
            {
                var lastWidth = Width;
                var lastHeight = Height;
                var lastRoi = Roi;
                using var mat = RawDecompressInternal();
                CompressInternal(mat);
                Width = lastWidth;
                Height = lastHeight;
                Roi = lastRoi;
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
    public Task<bool> ChangeCompressorAsync(MatCompressor compressor, CompressionLevel compressionLevel, bool reEncodeWithNewCompressor = false, CancellationToken cancellationToken = default)
    {
        return Task.Run(() => ChangeCompressor(compressor, compressionLevel, reEncodeWithNewCompressor), cancellationToken);
    }

    /// <summary>
    /// Changes the <see cref="Compressor"/> and optionally re-encodes the <see cref="Mat"/> with the new <paramref name="compressor"/> if the <see cref="Decompressor"/> is different from the set <paramref name="compressor"/>.
    /// </summary>
    /// <param name="compressor">New compressor</param>
    /// <param name="reEncodeWithNewCompressor">True to re-encodes the <see cref="Mat"/> with the new <see cref="Compressor"/>, otherwise false.</param>
    /// <param name="cancellationToken">A token to cancel the operation.</param>
    /// <returns>True if compressor has been changed, otherwise false.</returns>
    public Task<bool> ChangeCompressorAsync(MatCompressor compressor, bool reEncodeWithNewCompressor = false, CancellationToken cancellationToken = default)
    {
        return ChangeCompressorAsync(compressor, CompressionLevel, reEncodeWithNewCompressor, cancellationToken);
    }

    /// <summary>
    /// Sets the <see cref="CompressedBytes"/> to an empty byte array and sets <see cref="IsCompressed"/> to false.
    /// </summary>
    public void SetEmptyCompressedBytes()
    {
        _rwLock.EnterWriteLock();
        try
        {
            if (IsEmpty) return; // Already empty
            CompressedBytes = [];
            Roi = Rectangle.Empty;
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
            if (!IsEmpty)
            {
                CompressedBytes = [];
                Roi = Rectangle.Empty;
            }
            IsInitialized = isInitialized;
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
        _rwLock.EnterWriteLock();
        try
        {
            if (!IsEmpty)
            {
                CompressedBytes = [];
                Roi = Rectangle.Empty;
            }
            Width = src.Width;
            Height = src.Height;
            Depth = src.Depth;
            Channels = src.NumberOfChannels;
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
        _rwLock.EnterWriteLock();
        try
        {
            if (!IsEmpty)
            {
                CompressedBytes = [];
                Roi = Rectangle.Empty;
            }
            Width = src.Width;
            Height = src.Height;
            Depth = src.Depth;
            Channels = src.NumberOfChannels;
            IsInitialized = isInitialized;
        }
        finally
        {
            _rwLock.ExitWriteLock();
        }
    }

    /// <summary>
    /// Sets the <see cref="CompressedBytes"/> and <see cref="Compressor"/> and <see cref="Decompressor"/>.
    /// </summary>
    /// <param name="compressedBytes">The compressed byte array to set.</param>
    /// <param name="decompressor">The decompressor that matches the compressed data.</param>
    /// <param name="setCompressor">If <see langword="true"/>, also sets the <see cref="Compressor"/> to the specified decompressor.</param>
    public void SetCompressedBytes(byte[] compressedBytes, MatCompressor decompressor, bool setCompressor = true)
    {
        _rwLock.EnterWriteLock();
        try
        {
            CompressedBytes = compressedBytes;
            if (setCompressor) Compressor = decompressor;
            Decompressor = decompressor;
            if (ReferenceEquals(decompressor, MatCompressorNone.Instance)) IsCompressed = false;
        }
        finally
        {
            _rwLock.ExitWriteLock();
        }
    }

    /// <summary>
    /// Sets the <see cref="CompressedBytes"/> to uncompressed bitmap data.
    /// </summary>
    /// <param name="src">The source Mat whose raw data will be stored uncompressed.</param>
    private void SetUncompressed(Mat src)
    {
        CompressedBytes = src.ToArray();
        IsCompressed = false;
        Decompressor = MatCompressorNone.Instance;
    }

    /// <summary>
    /// Compresses the <see cref="Mat"/> into a byte array.
    /// </summary>
    /// <param name="src">The Mat to compress.</param>
    public void Compress(Mat src)
    {
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
        Width = src.Width;
        Height = src.Height;
        Depth = src.Depth;
        Channels = src.NumberOfChannels;
        Roi = Rectangle.Empty;

        if (src.IsEmpty)
        {
            CompressedBytes = [];
            return;
        }

        var srcLength = src.LengthInt32;
        if (srcLength <= ThresholdToCompress) // Do not compress if the size is smaller or equal to the threshold
        {
            SetUncompressed(src);
            return;
        }

        try
        {
            var compressed = Compressor.Compress(src, CompressionLevel);
            if (compressed.Length < srcLength) // Compressed ok
            {
                CompressedBytes = compressed;
                Decompressor = Compressor;
            }
            else // Compressed size is larger or equal to uncompressed size, store raw
            {
                SetUncompressed(src);
            }
        }
        catch (Exception ex) when (ex is not OutOfMemoryException and not AccessViolationException) // Cannot compress due some error
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
        _rwLock.EnterWriteLock();
        try
        {
            if (src.Roi.Size.IsEmpty)
            {
                Width = src.SourceMat.Width;
                Height = src.SourceMat.Height;
                Depth = src.SourceMat.Depth;
                Channels = src.SourceMat.NumberOfChannels;
                Roi = Rectangle.Empty;
                CompressedBytes = [];
                return;
            }

            if (src.IsSourceSameSizeOfRoi)
            {
                CompressInternal(src.SourceMat);
            }
            else
            {
                CompressInternal(src.RoiMat);
                Width = src.SourceMat.Width;
                Height = src.SourceMat.Height;
                Roi = src.Roi;
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
        if (IsEmpty) return Roi.Size.IsEmpty ? CreateMatZeros() : EmguExtensions.InitMat(Roi.Size, Channels, Depth);

        var mat = Roi.Size.IsEmpty ? CreateMat() : new Mat(Roi.Size, Depth, Channels);

        if (IsCompressed)
        {
            Decompressor.Decompress(CompressedBytes, mat);
        }
        else
        {
            mat.SetTo(CompressedBytes);
        }

        return mat;
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
    /// Decompresses the <see cref="CompressedBytes"/> into a new <see cref="Mat"/>.
    /// </summary>
    /// <returns></returns>
    public Mat Decompress()
    {
        _rwLock.EnterReadLock();
        try
        {
            if (IsEmpty) return CreateMatZeros();

            var mat = RawDecompressInternal();
            if (Roi.Size.IsEmpty) return mat;

            var fullMat = CreateMatZeros();
            try
            {
                using var roi = new Mat(fullMat, Roi);
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
        if (Width <= 0 || Height <= 0) return new Mat();
        return new Mat(Size, Depth, Channels);
    }

    /// <summary>
    /// Create a new <see cref="Mat"/> with the same size, depth, and channels as the <see cref="CMat"/> but with all bytes set to 0.
    /// </summary>
    /// <returns></returns>
    public Mat CreateMatZeros()
    {
        return EmguExtensions.InitMat(Size, Channels, Depth);
    }

    #endregion

    #region Copy and Clone
    /// <summary>
    /// Copies the <see cref="CMat"/> to the <paramref name="dst"/>.
    /// </summary>
    /// <param name="dst"></param>
    public void CopyTo(CMat dst)
    {
        if (ReferenceEquals(this, dst)) return;

        byte[] compressedBytes;
        ulong? hash;
        bool isInitialized;
        bool isCompressed;
        int thresholdToCompress;
        CompressionLevel compressionLevel;
        MatCompressor compressor;
        MatCompressor decompressor;
        int width;
        int height;
        DepthType depth;
        int channels;
        Rectangle roi;

        _rwLock.EnterReadLock();
        try
        {
            compressedBytes = _compressedBytes.ToArrayPerf();
            hash = _hash;
            isInitialized = IsInitialized;
            isCompressed = IsCompressed;
            thresholdToCompress = ThresholdToCompress;
            compressionLevel = CompressionLevel;
            compressor = Compressor;
            decompressor = Decompressor;
            width = Width;
            height = Height;
            depth = Depth;
            channels = Channels;
            roi = Roi;
        }
        finally
        {
            _rwLock.ExitReadLock();
        }

        dst._rwLock.EnterWriteLock();
        try
        {
            dst._compressedBytes = compressedBytes;
            dst._hash = hash;
            dst.IsInitialized = isInitialized;
            dst.IsCompressed = isCompressed;
            dst.ThresholdToCompress = thresholdToCompress;
            dst.CompressionLevel = compressionLevel;
            dst.Compressor = compressor;
            dst.Decompressor = decompressor;
            dst.Width = width;
            dst.Height = height;
            dst.Depth = depth;
            dst.Channels = channels;
            dst.Roi = roi;
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
        return $"{nameof(Decompressor)}: {Decompressor} @ {CompressionLevel}, {nameof(Size)}: {Size}, {nameof(UncompressedLength)}: {UncompressedLength}, {nameof(CompressedLength)}: {CompressedLength}, {nameof(IsCompressed)}: {IsCompressed}, {nameof(CompressionRatio)}: {CompressionRatio}x, {nameof(CompressionPercentage)}: {CompressionPercentage}%";
    }

    #endregion

    #region Equality

    /// <inheritdoc />
    public bool Equals(CMat? other)
    {
        if (ReferenceEquals(null, other)) return false;
        if (ReferenceEquals(this, other)) return true;
        return IsInitialized == other.IsInitialized
               && IsCompressed == other.IsCompressed
               && Width == other.Width
               && Height == other.Height
               && Depth == other.Depth
               && Channels == other.Channels
               && Roi.Equals(other.Roi)
               && CompressedLength == other.CompressedLength
               && Hash == other.Hash;
    }

    /// <inheritdoc />
    public override bool Equals(object? obj)
    {
        if (ReferenceEquals(null, obj)) return false;
        if (ReferenceEquals(this, obj)) return true;
        if (obj.GetType() != this.GetType()) return false;
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

    /// <inheritdoc />
    public override int GetHashCode() => Hash.GetHashCode();

    #endregion
}
