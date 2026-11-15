using System.Drawing;
using System.IO.Compression;

namespace EmguExtensions.Tests;

public class UnitTestCMatBuffers
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SharedCopy_ReplacementsDoNotChangeOtherInstance(bool useCopyTo)
    {
        var source = new CMat(3, 1) { ThresholdToCompress = 42 };
        byte[] bytes = [1, 2, 3];
        source.SetCompressedBytes(bytes, MatCompressorNone.Instance);
        var copy = useCopyTo ? new CMat() : source.Clone(true);
        if (useCopyTo)
            source.CopyTo(copy, true);

        Assert.Same(bytes, copy.CompressedBytes);
        Assert.Equal(source, copy);
        Assert.Equal(42, copy.ThresholdToCompress);
        source.SetCompressedBytes([4, 5, 6], MatCompressorNone.Instance);
        Assert.Same(bytes, copy.CompressedBytes);
        var sourceBytes = source.CompressedBytes;
        copy.SetEmptyCompressedBytes();
        Assert.Same(sourceBytes, source.CompressedBytes);
        Assert.NotEqual(source, copy);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void DetachAttach_RoundTripsRoiAndCompressionState(bool compressed)
    {
        using var mat = EmguCvExtensions.InitMat(new Size(100, 80));
        using var roi = new MatRoi(mat, new Rectangle(10, 20, 50, 30), true);
        var source = new CMat(
            roi,
            compressed ? MatCompressorGZip.Instance : MatCompressorNone.Instance,
            CompressionLevel.SmallestSize
        );
        var original = source.Clone();
        using var expected = source.Decompress();
        var hashCode = source.GetHashCode();

        var bytes = source.DetachCompressedBytes();
        Assert.Empty(source.CompressedBytes);
        Assert.Equal(original, source);
        Assert.Equal(hashCode, source.GetHashCode());
        Assert.Equal(original.Roi, source.Roi);
        Assert.Equal(original.IsInitialized, source.IsInitialized);
        Assert.Equal(original.IsCompressed, source.IsCompressed);
        Assert.Same(original.Decompressor, source.Decompressor);
        Assert.Throws<InvalidOperationException>(() => source.Decompress());
        Assert.Throws<InvalidOperationException>(() => source.RawDecompress());
        Assert.Throws<InvalidOperationException>(() => source.DetachCompressedBytes());
        Assert.Throws<InvalidOperationException>(() =>
            source.ChangeCompressor(MatCompressorBrotli.Instance, true)
        );

        var loadedBytes = bytes.ToArray();
        source.AttachCompressedBytes(loadedBytes);
        Assert.Same(loadedBytes, source.CompressedBytes);
        Assert.Equal(original, source);
        using var actual = source.Decompress();
        Assert.Equal(expected.ToArray(), actual.ToArray());
    }

    [Fact]
    public void TryDetach_RejectsReplacedBufferAndCachesHash()
    {
        var source = new CMat(3, 1);
        byte[] stale = [1, 2, 3];
        byte[] current = [4, 5, 6];
        source.SetCompressedBytes(stale, MatCompressorNone.Instance);
        source.SetCompressedBytes(current, MatCompressorNone.Instance);
        Assert.False(source.TryDetachCompressedBytes(stale));
        Assert.Same(current, source.CompressedBytes);
        var original = source.Clone(); // Neither hash is cached yet.
        Assert.True(source.TryDetachCompressedBytes(current));
        Assert.Equal(original.Hash, source.Hash);
        Assert.False(source.TryDetachCompressedBytes(source.CompressedBytes));
    }

    [Fact]
    public void DetachedEquality_DistinguishesDifferentContentAndCopiesDetachedState()
    {
        var left = new CMat(3, 1);
        var right = new CMat(3, 1);
        left.SetCompressedBytes([1, 2, 3], MatCompressorNone.Instance);
        right.SetCompressedBytes([4, 5, 6], MatCompressorNone.Instance);
        var bytes = left.DetachCompressedBytes();
        right.DetachCompressedBytes();
        Assert.NotEqual(left, right);
        var clone = left.Clone();
        Assert.Equal(left, clone);
        clone.AttachCompressedBytes(bytes.ToArray());
        Assert.Equal(left, clone);
        Assert.Empty(left.CompressedBytes);
    }

    [Fact]
    public void Attach_RejectsWrongContentAndCannotOverwriteReplacement()
    {
        var source = new CMat(3, 1);
        source.SetCompressedBytes([1, 2, 3], MatCompressorNone.Instance);
        Assert.Throws<InvalidOperationException>(() => source.AttachCompressedBytes([1, 2, 3]));
        var bytes = source.DetachCompressedBytes();
        Assert.Throws<ArgumentException>(() => source.AttachCompressedBytes([1]));
        Assert.Throws<ArgumentException>(() => source.AttachCompressedBytes([4, 5, 6]));
        source.SetCompressedBytes([7, 8, 9], MatCompressorNone.Instance);
        Assert.Throws<InvalidOperationException>(() => source.AttachCompressedBytes(bytes));
        Assert.Equal(new byte[] { 7, 8, 9 }, source.CompressedBytes);
    }

    [Fact]
    public void ClearDetachedBuffer_ResetsHashAndCompressionState()
    {
        var source = new CMat(3, 1);
        source.SetCompressedBytes([1, 2, 3], MatCompressorGZip.Instance);
        var bytes = source.DetachCompressedBytes();
        source.SetEmptyCompressedBytes();
        var empty = new CMat(3, 1);
        empty.SetEmptyCompressedBytes(true);
        Assert.False(source.IsCompressed);
        Assert.Equal(empty.Hash, source.Hash);
        Assert.Throws<InvalidOperationException>(() => source.AttachCompressedBytes(bytes));
    }

    [Fact]
    public void TryDetach_ConcurrentAttempts_OnlyOneSucceeds()
    {
        var source = new CMat(3, 1);
        byte[] bytes = [1, 2, 3];
        source.SetCompressedBytes(bytes, MatCompressorNone.Instance);
        var successes = 0;
        Parallel.For(0, 32, _ =>
        {
            if (source.TryDetachCompressedBytes(bytes))
                Interlocked.Increment(ref successes);
        });
        Assert.Equal(1, successes);
        source.AttachCompressedBytes(bytes);
        Assert.Same(bytes, source.CompressedBytes);
    }

    [Fact]
    public void EmptyBuffer_RoundTripPreservesUninitializedState()
    {
        var source = new CMat(3, 1);
        var original = source.Clone();
        var bytes = source.DetachCompressedBytes();
        Assert.Empty(bytes);
        Assert.Equal(original, source);
        source.AttachCompressedBytes(bytes);
        Assert.False(source.IsInitialized);
        Assert.Equal(original, source);
    }
}
