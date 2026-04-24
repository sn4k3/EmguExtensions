using System.Drawing;
using System.IO.Compression;
using Emgu.CV;
using Emgu.CV.CvEnum;

namespace EmguExtensions.Tests;

/// <summary>
/// Comprehensive tests for <see cref="CMat"/>.
/// </summary>
/// <remarks>
/// Design note: <c>CMat.Mat { get; }</c> allocates a new <see cref="Mat"/> on every access via
/// <see cref="CMat.Decompress"/>. Callers are responsible for disposing the returned Mat.
/// No test exercises the <c>Mat</c> property directly to avoid this footgun — use
/// <see cref="CMat.Decompress"/> explicitly instead.
/// </remarks>
public class UnitTestCMat
{
    // 100×80: top half black (0), bottom half white (255). ~8 000 bytes, well above the 512 threshold.
    private static Mat CreateLargeMat()
    {
        var mat = EmguExtensions.InitMat(new Size(100, 80));
        CvInvoke.Rectangle(mat, new Rectangle(0, 40, 100, 40), EmguExtensions.WhiteColor, -1);
        return mat;
    }

    // 10×10 single-channel = 100 bytes, well below the 512 default threshold.
    private static Mat CreateSmallMat()
    {
        var mat = EmguExtensions.InitMat(new Size(10, 10));
        CvInvoke.Rectangle(mat, new Rectangle(0, 5, 10, 5), EmguExtensions.WhiteColor, -1);
        return mat;
    }

    // Asserts that two Mats contain identical pixel data.
    private static void AssertMatsEqual(Mat expected, Mat actual)
    {
        Assert.Equal(expected.Size, actual.Size);
        Assert.Equal(expected.Depth, actual.Depth);
        Assert.Equal(expected.NumberOfChannels, actual.NumberOfChannels);
        Assert.Equal(expected.ToArray(), actual.ToArray());
    }

    #region Constructors

    [Fact]
    public void Constructor_Default_AllDefaultsAndNotInitialized()
    {
        var cmat = new CMat();

        Assert.False(cmat.IsInitialized);
        Assert.False(cmat.IsCompressed);
        Assert.True(cmat.IsEmpty);
        Assert.Equal(0, cmat.Width);
        Assert.Equal(0, cmat.Height);
        Assert.Equal(1, cmat.Channels);
        Assert.Equal(DepthType.Cv8U, cmat.Depth);
    }

    [Fact]
    public void Constructor_WithDimensions_SetsMetadataButNotInitialized()
    {
        var cmat = new CMat(50, 40, DepthType.Cv16U, 3);

        Assert.Equal(50, cmat.Width);
        Assert.Equal(40, cmat.Height);
        Assert.Equal(DepthType.Cv16U, cmat.Depth);
        Assert.Equal(3, cmat.Channels);
        Assert.False(cmat.IsInitialized);
        Assert.True(cmat.IsEmpty);
    }

    [Fact]
    public void Constructor_WithCompressorAndDimensions_SetsCompressorAndDecompressor()
    {
        var cmat = new CMat(MatCompressorGZip.Instance, 100, 80);

        Assert.Same(MatCompressorGZip.Instance, cmat.Compressor);
        Assert.Same(MatCompressorGZip.Instance, cmat.Decompressor);
        Assert.False(cmat.IsInitialized);
    }

    [Fact]
    public void Constructor_WithMat_CompressesAndIsInitialized()
    {
        using var mat = CreateLargeMat();

        var cmat = new CMat(mat);

        Assert.True(cmat.IsInitialized);
        Assert.False(cmat.IsEmpty);
        Assert.Equal(mat.Width, cmat.Width);
        Assert.Equal(mat.Height, cmat.Height);
        Assert.Equal(mat.Depth, cmat.Depth);
        Assert.Equal(mat.NumberOfChannels, cmat.Channels);
        Assert.Equal(Rectangle.Empty, cmat.Roi);
    }

    [Fact]
    public void Constructor_WithEmptyMat_IsInitializedButEmpty()
    {
        using var emptyMat = new Mat();

        var cmat = new CMat(emptyMat);

        Assert.True(cmat.IsInitialized);
        Assert.True(cmat.IsEmpty);
        Assert.False(cmat.IsCompressed);
    }

    [Fact]
    public void Constructor_WithMatRoi_SubRegion_StoresRoiAndSourceDimensions()
    {
        using var mat = CreateLargeMat();
        var roiRect = new Rectangle(10, 10, 50, 30);
        using var matRoi = new MatRoi(mat, roiRect, leaveOpen: true);

        var cmat = new CMat(matRoi);

        Assert.Equal(mat.Width, cmat.Width);
        Assert.Equal(mat.Height, cmat.Height);
        Assert.Equal(roiRect, cmat.Roi);
        Assert.True(cmat.IsInitialized);
        Assert.False(cmat.IsEmpty);
    }

    [Fact]
    public void Constructor_WithMatRoi_FullSizeSource_SetsRoiToEmpty()
    {
        using var mat = CreateLargeMat();
        var fullRoi = new Rectangle(0, 0, mat.Width, mat.Height);
        using var matRoi = new MatRoi(mat, fullRoi, leaveOpen: true);

        var cmat = new CMat(matRoi);

        // When IsSourceSameSizeOfRoi, delegates to Compress(SourceMat) which sets Roi = Empty
        Assert.Equal(Rectangle.Empty, cmat.Roi);
        Assert.Equal(mat.Width, cmat.Width);
        Assert.Equal(mat.Height, cmat.Height);
    }

    #endregion

    #region Compress(Mat)

    [Fact]
    public void Compress_LargeUniformMat_IsCompressedTrueDecompressorMatchesCompressor()
    {
        using var mat = CreateLargeMat();
        var cmat = new CMat(MatCompressorGZip.Instance);

        cmat.Compress(mat);

        Assert.True(cmat.IsInitialized);
        Assert.True(cmat.IsCompressed);
        Assert.Same(MatCompressorGZip.Instance, cmat.Decompressor);
    }

    [Fact]
    public void Compress_MatBelowThreshold_IsCompressedFalseAndDecompressorIsNone()
    {
        // 10×10 = 100 bytes < 512 default threshold → stored raw
        using var mat = CreateSmallMat();
        var cmat = new CMat(MatCompressorGZip.Instance);

        cmat.Compress(mat);

        Assert.True(cmat.IsInitialized);
        Assert.False(cmat.IsCompressed);
        Assert.Same(MatCompressorNone.Instance, cmat.Decompressor);
        Assert.Equal(100, cmat.CompressedLength); // raw bytes
    }

    [Fact]
    public void Compress_EmptyMat_IsInitializedTrueButEmpty()
    {
        using var empty = new Mat();
        var cmat = new CMat();

        cmat.Compress(empty);

        Assert.True(cmat.IsInitialized);
        Assert.True(cmat.IsEmpty);
        Assert.False(cmat.IsCompressed);
    }

    [Fact]
    public void Compress_SetsMatDimensionsAndClearsRoi()
    {
        using var mat = CreateLargeMat();
        using var matRoi = new Mat(mat, new Rectangle(1, 1, 10, 10));
        var cmat = new CMat(matRoi);

        cmat.Compress(mat);

        Assert.Equal(mat.Width, cmat.Width);
        Assert.Equal(mat.Height, cmat.Height);
        Assert.Equal(mat.Depth, cmat.Depth);
        Assert.Equal(mat.NumberOfChannels, cmat.Channels);
        Assert.Equal(Rectangle.Empty, cmat.Roi);
    }

    [Fact]
    public void Compress_CompressedSizeLargerThanRaw_FallsBackToUncompressed()
    {
        // Force Brotli to produce output larger than input by setting a very high threshold and a 1-channel 1×1 mat
        using var mat = EmguExtensions.InitMat(new Size(1, 1));
        var cmat = new CMat(MatCompressorBrotli.Instance) { ThresholdToCompress = 0 };

        cmat.Compress(mat);

        // Brotli output for 1 byte is almost certainly larger — fallback to raw
        Assert.False(cmat.IsCompressed);
        Assert.Same(MatCompressorNone.Instance, cmat.Decompressor);
    }

    [Fact]
    public void Compress_OverwritesPreviousCompression()
    {
        using var mat1 = CreateLargeMat();
        using var mat2 = EmguExtensions.InitMat(new Size(50, 30)); // all black
        var cmat = new CMat(mat1);
        var originalLength = cmat.CompressedLength;

        cmat.Compress(mat2);

        Assert.Equal(50, cmat.Width);
        Assert.Equal(30, cmat.Height);
        Assert.NotEqual(originalLength, cmat.CompressedLength);
    }

    #endregion

    #region Compress(MatRoi)

    [Fact]
    public void Compress_MatRoiSubRegion_StoresRoiCoordinatesAndSourceSize()
    {
        using var mat = CreateLargeMat();
        var roiRect = new Rectangle(5, 10, 40, 30);
        using var matRoi = new MatRoi(mat, roiRect, leaveOpen: true);
        var cmat = new CMat();

        cmat.Compress(matRoi);

        Assert.Equal(mat.Width, cmat.Width);
        Assert.Equal(mat.Height, cmat.Height);
        Assert.Equal(roiRect, cmat.Roi);
        Assert.True(cmat.IsInitialized);
    }

    [Fact]
    public void Compress_MatRoiEmptyRoi_IsInitializedAndEmpty()
    {
        using var mat = CreateLargeMat();
        using var matRoi = new MatRoi(mat, Rectangle.Empty, leaveOpen: true);
        var cmat = new CMat();

        cmat.Compress(matRoi);

        Assert.True(cmat.IsInitialized);
        Assert.True(cmat.IsEmpty);
        Assert.Equal(mat.Width, cmat.Width);
        Assert.Equal(mat.Height, cmat.Height);
    }

    #endregion

    #region Decompress / RawDecompress

    [Theory]
    [MemberData(nameof(AllCompressors))]
    public void Decompress_AllCompressors_RoundTripIsPixelEqual(MatCompressor compressor)
    {
        using var original = CreateLargeMat();
        var cmat = new CMat(original, compressor);

        using var result = cmat.Decompress();

        AssertMatsEqual(original, result);
    }

    [Fact]
    public void Decompress_EmptyBytes_ReturnsZeroedMatOfCorrectSize()
    {
        var cmat = new CMat(100, 80);

        using var result = cmat.Decompress();

        Assert.Equal(new Size(100, 80), result.Size);
        Assert.All(result.ToArray(), b => Assert.Equal(0, b));
    }

    [Fact]
    public void Decompress_WithRoi_ReturnsFullSizeMatWithRoiDataPlaced()
    {
        using var mat = CreateLargeMat();
        var roiRect = new Rectangle(0, 40, 100, 40); // white bottom half
        using var matRoi = new MatRoi(mat, roiRect, leaveOpen: true);
        var cmat = new CMat(matRoi, MatCompressorGZip.Instance);

        using var result = cmat.Decompress();

        Assert.Equal(mat.Size, result.Size);
        // The white half should have been placed back at y=40
        var bytes = result.ToArray();
        // First 4000 bytes (top half) = black (0)
        Assert.All(bytes.Take(4000), b => Assert.Equal(0, b));
        // Last 4000 bytes (bottom half) = white (255)
        Assert.All(bytes.Skip(4000), b => Assert.Equal(255, b));
    }

    [Fact]
    public void RawDecompress_WithRoi_ReturnsRoiSizeNotFullSize()
    {
        using var mat = CreateLargeMat();
        var roiRect = new Rectangle(10, 10, 40, 30);
        using var matRoi = new MatRoi(mat, roiRect, leaveOpen: true);
        var cmat = new CMat(matRoi, MatCompressorGZip.Instance);

        using var raw = cmat.RawDecompress();

        Assert.Equal(roiRect.Size, raw.Size);
    }

    [Fact]
    public void RawDecompress_EmptyBytes_ReturnsZeroedMat()
    {
        var cmat = new CMat(60, 40);

        using var result = cmat.RawDecompress();

        Assert.Equal(new Size(60, 40), result.Size);
    }

    [Fact]
    public void Decompress_SmallMatBelowThreshold_RoundTripCorrect()
    {
        using var small = CreateSmallMat();
        var cmat = new CMat(small, MatCompressorBrotli.Instance);

        Assert.False(cmat.IsCompressed); // stored raw due to threshold

        using var result = cmat.Decompress();
        AssertMatsEqual(small, result);
    }

    #endregion

    #region ChangeCompressor

    [Fact]
    public void ChangeCompressor_SameCompressorSameLevel_ReturnsFalse()
    {
        using var mat = CreateLargeMat();
        var cmat = new CMat(mat, MatCompressorGZip.Instance, CompressionLevel.Optimal);

        var changed = cmat.ChangeCompressor(MatCompressorGZip.Instance, CompressionLevel.Optimal);

        Assert.False(changed);
    }

    [Fact]
    public void ChangeCompressor_DifferentCompressor_NoReEncode_UpdatesCompressorNotDecompressor()
    {
        using var mat = CreateLargeMat();
        var cmat = new CMat(mat, MatCompressorGZip.Instance);
        var originalDecompressor = cmat.Decompressor;
        var originalLength = cmat.CompressedLength;

        var changed = cmat.ChangeCompressor(MatCompressorBrotli.Instance, reEncodeWithNewCompressor: false);

        Assert.True(changed);
        Assert.Same(MatCompressorBrotli.Instance, cmat.Compressor);
        Assert.Same(originalDecompressor, cmat.Decompressor); // unchanged — bytes still encoded with original
        Assert.Equal(originalLength, cmat.CompressedLength);  // bytes untouched
    }

    [Fact]
    public void ChangeCompressor_WithReEncode_UpdatesDecompressorAndReEncodes()
    {
        using var mat = CreateLargeMat();
        var cmat = new CMat(mat, MatCompressorGZip.Instance);

        cmat.ChangeCompressor(MatCompressorBrotli.Instance, reEncodeWithNewCompressor: true);

        Assert.Same(MatCompressorBrotli.Instance, cmat.Compressor);
        Assert.Same(MatCompressorBrotli.Instance, cmat.Decompressor);
        // round-trip must still be valid
        using var result = cmat.Decompress();
        AssertMatsEqual(mat, result);
    }

    [Fact]
    public void ChangeCompressor_WithRoiReEncode_PreservesSourceDimensionsAndPlacement()
    {
        using var mat = CreateLargeMat();
        var roiRect = new Rectangle(0, 40, 100, 40);
        using var matRoi = new MatRoi(mat, roiRect, leaveOpen: true);
        var cmat = new CMat(matRoi, MatCompressorGZip.Instance);

        cmat.ChangeCompressor(MatCompressorBrotli.Instance, reEncodeWithNewCompressor: true);

        Assert.Equal(mat.Width, cmat.Width);
        Assert.Equal(mat.Height, cmat.Height);
        Assert.Equal(roiRect, cmat.Roi);

        using var result = cmat.Decompress();
        Assert.Equal(mat.Size, result.Size);
        Assert.Equal(mat.ToArray(), result.ToArray());
    }

    [Fact]
    public void ChangeCompressor_SmallestSize_ProducesSmallerOutputThanOptimal()
    {
        using var mat = CreateLargeMat();
        var cmat = new CMat(mat, MatCompressorGZip.Instance, CompressionLevel.Optimal);
        var optimalLength = cmat.CompressedLength;

        cmat.ChangeCompressor(MatCompressorGZip.Instance, CompressionLevel.SmallestSize, reEncodeWithNewCompressor: true);

        Assert.True(cmat.CompressedLength <= optimalLength,
            $"SmallestSize ({cmat.CompressedLength}) should be ≤ Optimal ({optimalLength})");
    }

    [Fact]
    public void ChangeCompressor_WhenEmpty_NoReEncodeEvenIfRequested()
    {
        var cmat = new CMat(MatCompressorGZip.Instance, 100, 80);

        var changed = cmat.ChangeCompressor(MatCompressorBrotli.Instance, reEncodeWithNewCompressor: true);

        Assert.True(changed); // compressor changed
        Assert.True(cmat.IsEmpty); // no bytes written
    }

    [Fact]
    public async Task ChangeCompressorAsync_WithReEncode_ReEncodesCorrectly()
    {
        using var mat = CreateLargeMat();
        var cmat = new CMat(mat, MatCompressorGZip.Instance);

        var changed = await cmat.ChangeCompressorAsync(MatCompressorBrotli.Instance, reEncodeWithNewCompressor: true, cancellationToken: TestContext.Current.CancellationToken);

        Assert.True(changed);
        Assert.Same(MatCompressorBrotli.Instance, cmat.Decompressor);
        using var result = await cmat.DecompressAsync(TestContext.Current.CancellationToken);
        AssertMatsEqual(mat, result);
    }

    #endregion

    #region SetEmptyCompressedBytes

    [Fact]
    public void SetEmptyCompressedBytes_WhenPopulated_ClearsAllBytesAndRoi()
    {
        using var mat = CreateLargeMat();
        var cmat = new CMat(mat);
        Assert.False(cmat.IsEmpty);

        cmat.SetEmptyCompressedBytes();

        Assert.True(cmat.IsEmpty);
        Assert.Equal(Rectangle.Empty, cmat.Roi);
        Assert.True(cmat.IsInitialized); // was true before, setter preserves it
    }

    [Fact]
    public void SetEmptyCompressedBytes_AlreadyEmpty_IsNoOp()
    {
        var cmat = new CMat(); // fresh, IsInitialized=false, IsEmpty=true

        cmat.SetEmptyCompressedBytes(); // must return early without calling setter

        Assert.False(cmat.IsInitialized); // setter was never called, so remains false
    }

    [Fact]
    public void SetEmptyCompressedBytes_WithIsInitializedFalse_OverridesInitializedState()
    {
        using var mat = CreateLargeMat();
        var cmat = new CMat(mat);
        Assert.True(cmat.IsInitialized);

        cmat.SetEmptyCompressedBytes(isInitialized: false);

        Assert.True(cmat.IsEmpty);
        Assert.False(cmat.IsInitialized);
    }

    [Fact]
    public void SetEmptyCompressedBytes_WithMat_ExtractsDimensionsFromMat()
    {
        using var mat = CreateLargeMat();
        var cmat = new CMat(mat); // populate first so it's not empty
        cmat.SetEmptyCompressedBytes(mat);

        Assert.Equal(mat.Width, cmat.Width);
        Assert.Equal(mat.Height, cmat.Height);
        Assert.Equal(mat.Depth, cmat.Depth);
        Assert.Equal(mat.NumberOfChannels, cmat.Channels);
        Assert.True(cmat.IsEmpty);
    }

    #endregion

    #region SetCompressedBytes

    [Fact]
    public void SetCompressedBytes_SetsCompressorAndDecompressor()
    {
        var cmat = new CMat();
        var bytes = new byte[] { 1, 2, 3, 4, 5 };

        cmat.SetCompressedBytes(bytes, MatCompressorGZip.Instance, setCompressor: true);

        Assert.Equal(bytes, cmat.CompressedBytes);
        Assert.Same(MatCompressorGZip.Instance, cmat.Compressor);
        Assert.Same(MatCompressorGZip.Instance, cmat.Decompressor);
        Assert.True(cmat.IsInitialized);
    }

    [Fact]
    public void SetCompressedBytes_WithSetCompressorFalse_OnlySetsDecompressor()
    {
        var cmat = new CMat { Compressor = MatCompressorBrotli.Instance };
        var bytes = new byte[] { 10, 20, 30 };

        cmat.SetCompressedBytes(bytes, MatCompressorGZip.Instance, setCompressor: false);

        Assert.Same(MatCompressorBrotli.Instance, cmat.Compressor);   // unchanged
        Assert.Same(MatCompressorGZip.Instance, cmat.Decompressor);   // updated
    }

    [Fact]
    public void SetCompressedBytes_WithNoneDecompressor_MarksBytesAsUncompressed()
    {
        var cmat = new CMat();
        var bytes = new byte[] { 10, 20, 30 };

        cmat.SetCompressedBytes(bytes, MatCompressorNone.Instance);

        Assert.False(cmat.IsCompressed);
        Assert.Same(MatCompressorNone.Instance, cmat.Compressor);
        Assert.Same(MatCompressorNone.Instance, cmat.Decompressor);
    }

    #endregion

    #region Clone / CopyTo

    [Fact]
    public void Clone_ProducesEqualCopy()
    {
        using var mat = CreateLargeMat();
        var original = new CMat(mat, MatCompressorGZip.Instance);

        var clone = original.Clone();

        Assert.Equal(original, clone);
        Assert.NotSame(original, clone);
        Assert.NotSame(original.CompressedBytes, clone.CompressedBytes);
    }

    [Fact]
    public void Clone_MutatingOriginalBytesDoesNotAffectClone()
    {
        using var mat = CreateLargeMat();
        var original = new CMat(mat, MatCompressorNone.Instance);
        var clone = original.Clone();
        var cloneLength = clone.CompressedLength;

        // Re-compress with a new mat to change original's bytes
        using var newMat = EmguExtensions.InitMat(new Size(50, 30));
        original.Compress(newMat);

        Assert.Equal(cloneLength, clone.CompressedLength); // clone unaffected
        Assert.NotEqual(original, clone);
    }

    [Fact]
    public void CopyTo_CopiesAllFields()
    {
        using var mat = CreateLargeMat();
        var src = new CMat(mat, MatCompressorGZip.Instance, CompressionLevel.SmallestSize)
        {
            ThresholdToCompress = 1024,
        };
        var dst = new CMat();

        src.CopyTo(dst);

        Assert.Equal(src.CompressedBytes, dst.CompressedBytes);
        Assert.Equal(src.IsInitialized, dst.IsInitialized);
        Assert.Equal(src.IsCompressed, dst.IsCompressed);
        Assert.Equal(src.Width, dst.Width);
        Assert.Equal(src.Height, dst.Height);
        Assert.Equal(src.Depth, dst.Depth);
        Assert.Equal(src.Channels, dst.Channels);
        Assert.Equal(src.Roi, dst.Roi);
        Assert.Equal(src.CompressionLevel, dst.CompressionLevel);
        Assert.Same(src.Compressor, dst.Compressor);
        Assert.Same(src.Decompressor, dst.Decompressor);
        Assert.Equal(src.ThresholdToCompress, dst.ThresholdToCompress);
    }

    [Fact]
    public void CopyTo_SameInstance_IsNoOp()
    {
        using var mat = CreateLargeMat();
        var cmat = new CMat(mat, MatCompressorGZip.Instance);
        var hash = cmat.Hash;

        var exception = Record.Exception(() => cmat.CopyTo(cmat));

        Assert.Null(exception);
        Assert.Equal(hash, cmat.Hash);
    }

    #endregion

    #region Equality

    [Fact]
    public void Equals_SameInstance_ReturnsTrue()
    {
        using var mat = CreateLargeMat();
        var cmat = new CMat(mat);

        Assert.Equal(cmat, cmat);
    }

    [Fact]
    public void Equals_Clone_ReturnsTrue()
    {
        using var mat = CreateLargeMat();
        var cmat = new CMat(mat, MatCompressorBrotli.Instance);
        var clone = cmat.Clone();

        Assert.Equal(cmat, clone);
    }

    [Fact]
    public void Equals_CompressorChangedWithoutReEncode_StillEqual()
    {
        using var mat = CreateLargeMat();
        var cmat = new CMat(mat, MatCompressorBrotli.Instance);
        var clone = cmat.Clone();

        // Changing Compressor on clone doesn't change bytes, so Equals still true
        clone.ChangeCompressor(MatCompressorGZip.Instance, reEncodeWithNewCompressor: false);

        Assert.Equal(cmat, clone);
    }

    [Fact]
    public void Equals_ReEncodedWithDifferentCompressor_NotEqual()
    {
        using var mat = CreateLargeMat();
        var cmat = new CMat(mat, MatCompressorBrotli.Instance);
        var clone = cmat.Clone();

        clone.ChangeCompressor(MatCompressorGZip.Instance, reEncodeWithNewCompressor: true);

        Assert.NotEqual(cmat, clone);
    }

    [Fact]
    public void Equals_Null_ReturnsFalse()
    {
        using var mat = CreateLargeMat();
        var cmat = new CMat(mat);

        Assert.False(cmat.Equals(null));
        Assert.False(cmat == null);
        Assert.True(cmat != null);
    }

    [Fact]
    public void Equals_BothNull_ViaOperator_ReturnsTrue()
    {
        CMat? a = null;
        CMat? b = null;

        Assert.True(a == b);
    }

    [Fact]
    public void GetHashCode_SameBytes_SameHash()
    {
        using var mat = CreateLargeMat();
        var c1 = new CMat(mat, MatCompressorGZip.Instance);
        var c2 = c1.Clone();

        Assert.Equal(c1.GetHashCode(), c2.GetHashCode());
        Assert.Equal(c1.Hash, c2.Hash);
    }

    [Fact]
    public void GetHashCode_DifferentBytes_DifferentHash()
    {
        using var mat1 = CreateLargeMat();
        using var mat2 = EmguExtensions.InitMat(new Size(100, 80)); // all black
        var c1 = new CMat(mat1, MatCompressorNone.Instance) { ThresholdToCompress = 0 };
        var c2 = new CMat(mat2, MatCompressorNone.Instance) { ThresholdToCompress = 0 };

        Assert.NotEqual(c1.Hash, c2.Hash);
    }

    #endregion

    #region Metrics

    [Fact]
    public void CompressionRatio_WhenNotCompressed_ReturnsZero()
    {
        var cmat = new CMat(100, 80);

        Assert.Equal(0f, cmat.CompressionRatio);
    }

    [Fact]
    public void CompressionRatio_CompressedMat_GreaterThanOne()
    {
        using var mat = CreateLargeMat();
        var cmat = new CMat(mat, MatCompressorGZip.Instance);

        Assert.True(cmat.IsCompressed);
        Assert.True(cmat.CompressionRatio > 1f,
            $"Expected ratio > 1, got {cmat.CompressionRatio}");
    }

    [Fact]
    public void CompressionPercentage_EmptyBytes_ReturnsZero()
    {
        var cmat = new CMat(100, 80);

        Assert.Equal(0f, cmat.CompressionPercentage);
    }

    [Fact]
    public void SavedBytes_CompressedMat_PositiveValue()
    {
        using var mat = CreateLargeMat();
        var cmat = new CMat(mat, MatCompressorGZip.Instance);

        Assert.True(cmat.SavedBytes > 0,
            $"Expected positive saved bytes, got {cmat.SavedBytes}");
        Assert.Equal(cmat.UncompressedLength - cmat.CompressedLength, cmat.SavedBytes);
    }

    [Fact]
    public void UncompressedLength_WithRoi_UsesRoiDimensions()
    {
        using var mat = CreateLargeMat();
        var roiRect = new Rectangle(10, 10, 40, 30);
        using var matRoi = new MatRoi(mat, roiRect, leaveOpen: true);
        var cmat = new CMat(matRoi);

        // UncompressedLength for ROI should be 40*30 = 1200
        Assert.Equal(40 * 30, cmat.UncompressedLength);
    }

    [Fact]
    public void UncompressedLength_WithoutRoi_UsesFullDimensions()
    {
        using var mat = CreateLargeMat();
        var cmat = new CMat(mat);

        Assert.Equal(100 * 80, cmat.UncompressedLength);
    }

    #endregion

    #region CreateMat / CreateMatZeros

    [Fact]
    public void CreateMat_ZeroDimensions_ReturnsEmptyMat()
    {
        var cmat = new CMat();

        using var result = cmat.CreateMat();

        Assert.True(result.IsEmpty);
    }

    [Fact]
    public void CreateMat_WithDimensions_ReturnsCorrectSizeAndType()
    {
        var cmat = new CMat(60, 40, DepthType.Cv8U, 1);

        using var result = cmat.CreateMat();

        Assert.Equal(new Size(60, 40), result.Size);
        Assert.Equal(DepthType.Cv8U, result.Depth);
        Assert.Equal(1, result.NumberOfChannels);
    }

    [Fact]
    public void CreateMatZeros_ReturnsZeroedMat()
    {
        var cmat = new CMat(20, 15);

        using var result = cmat.CreateMatZeros();

        Assert.Equal(new Size(20, 15), result.Size);
        Assert.All(result.ToArray(), b => Assert.Equal(0, b));
    }

    #endregion

    #region Test data

    public static TheoryData<MatCompressor> AllCompressors()
    {
        var data = new TheoryData<MatCompressor>();
        foreach (var c in MatCompressor.AvailableCompressors)
            data.Add(c);
        return data;
    }

    #endregion
}
