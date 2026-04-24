using System.Collections.Concurrent;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace EmguExtensions.Tests;

/// <summary>
/// Thread-safety tests for <see cref="CMat"/>.
/// </summary>
/// <remarks>
/// <see cref="CMat"/> uses a <see cref="ReaderWriterLockSlim"/> for multiple-reader/single-writer
/// concurrency. Write operations (<see cref="CMat.Compress(Mat)"/>, <see cref="CMat.SetEmptyCompressedBytes()"/>,
/// <see cref="CMat.SetCompressedBytes"/>, <see cref="CMat.ChangeCompressor(MatCompressor, bool)"/>)
/// acquire a write lock; read operations (<see cref="CMat.Decompress"/>, <see cref="CMat.RawDecompress"/>,
/// <see cref="CMat.CopyTo"/>, <see cref="CMat.Hash"/>) acquire a read lock.
/// </remarks>
public class UnitTestCMatThreadSafety
{
    private const int ThreadCount = 8;
    private static readonly TimeSpan TestTimeout = TimeSpan.FromSeconds(10);

    private static Mat CreateMat(int value = 128)
    {
        var mat = EmguExtensions.InitMat(new Size(100, 80));
        mat.SetTo(new MCvScalar(value));
        return mat;
    }

    // -------------------------------------------------------------------------
    // Concurrent Compress(Mat)
    // -------------------------------------------------------------------------

    [Fact]
    public async Task Compress_CalledConcurrently_NoExceptionAndStateIsConsistent()
    {
        var cmat = new CMat();
        var barrier = new Barrier(ThreadCount);

        var tasks = Enumerable.Range(0, ThreadCount).Select(i => Task.Run(() =>
        {
            using var mat = CreateMat(i * 10);
            barrier.SignalAndWait(TestContext.Current.CancellationToken);
            cmat.Compress(mat);
        }, TestContext.Current.CancellationToken)).ToArray();

        await Task.WhenAll(tasks).WaitAsync(TestTimeout, TestContext.Current.CancellationToken);

        Assert.True(cmat.IsInitialized);
        Assert.Equal(100, cmat.Width);
        Assert.Equal(80, cmat.Height);
        Assert.Equal(DepthType.Cv8U, cmat.Depth);
        Assert.Equal(1, cmat.Channels);
        Assert.False(cmat.IsEmpty);
    }

    [Fact]
    public async Task Compress_CalledConcurrentlyWithSameSource_DecompressAfterAllWritesProducesValidMat()
    {
        // All threads write the same pixel value so any winner produces the same result.
        const int pixelValue = 200;
        var expected = CreateMat(pixelValue).ToArray();

        var cmat = new CMat();
        var barrier = new Barrier(ThreadCount);

        var tasks = Enumerable.Range(0, ThreadCount).Select(_ => Task.Run(() =>
        {
            using var mat = CreateMat(pixelValue);
            barrier.SignalAndWait(TestContext.Current.CancellationToken);
            cmat.Compress(mat);
        }, TestContext.Current.CancellationToken)).ToArray();

        await Task.WhenAll(tasks).WaitAsync(TestTimeout, TestContext.Current.CancellationToken);

        using var result = await cmat.DecompressAsync(TestContext.Current.CancellationToken);
        Assert.Equal(new Size(100, 80), result.Size);
        Assert.Equal(expected, result.ToArray());
    }

    [Fact]
    public async Task Compress_CalledConcurrentlyWithMixOfEmptyAndNonEmpty_NoException()
    {
        var cmat = new CMat();
        var barrier = new Barrier(ThreadCount);

        var tasks = Enumerable.Range(0, ThreadCount).Select(i => Task.Run(() =>
        {
            using var mat = i % 2 == 0 ? new Mat() : CreateMat();
            barrier.SignalAndWait(TestContext.Current.CancellationToken);
            cmat.Compress(mat);
        }, TestContext.Current.CancellationToken)).ToArray();

        await Task.WhenAll(tasks).WaitAsync(TestTimeout, TestContext.Current.CancellationToken);

        Assert.True(cmat.IsInitialized);
    }

    // -------------------------------------------------------------------------
    // Concurrent CompressAsync(Mat)
    // -------------------------------------------------------------------------

    [Fact]
    public async Task CompressAsync_CalledConcurrently_NoExceptionAndStateIsConsistent()
    {
        var cmat = new CMat();

        var tasks = Enumerable.Range(0, ThreadCount).Select(i => Task.Run(async () =>
        {
            using var mat = CreateMat(i * 15);
            await cmat.CompressAsync(mat, TestContext.Current.CancellationToken);
        }, TestContext.Current.CancellationToken)).ToArray();

        await Task.WhenAll(tasks).WaitAsync(TestTimeout, TestContext.Current.CancellationToken);

        Assert.True(cmat.IsInitialized);
        Assert.Equal(100, cmat.Width);
        Assert.Equal(80, cmat.Height);
        Assert.False(cmat.IsEmpty);
    }

    // -------------------------------------------------------------------------
    // Compress(MatRoi) — no longer recursive, uses CompressInternal
    // -------------------------------------------------------------------------

    [Fact]
    public async Task Compress_MatRoiNonFullSource_DoesNotDeadlock()
    {
        // !IsSourceSameSizeOfRoi path: Compress(MatRoi) → CompressInternal(RoiMat) — no recursive lock entry.
        using var sourceMat = CreateMat(64);
        var roiRect = new Rectangle(10, 10, 50, 40);
        using var matRoi = new MatRoi(sourceMat, roiRect, leaveOpen: true);

        var cmat = new CMat();

        await Task.Run(() => cmat.Compress(matRoi), TestContext.Current.CancellationToken)
            .WaitAsync(TestTimeout, TestContext.Current.CancellationToken);

        Assert.True(cmat.IsInitialized);
        Assert.Equal(100, cmat.Width);
        Assert.Equal(80, cmat.Height);
        Assert.Equal(roiRect, cmat.Roi);
    }

    [Fact]
    public async Task Compress_MatRoiFullSource_DoesNotDeadlock()
    {
        // IsSourceSameSizeOfRoi path: Compress(MatRoi) → CompressInternal(SourceMat) — no recursive lock entry.
        using var sourceMat = CreateMat(32);
        var fullRoi = new Rectangle(0, 0, sourceMat.Width, sourceMat.Height);
        using var matRoi = new MatRoi(sourceMat, fullRoi, leaveOpen: true);

        var cmat = new CMat();

        await Task.Run(() => cmat.Compress(matRoi), TestContext.Current.CancellationToken)
            .WaitAsync(TestTimeout, TestContext.Current.CancellationToken);

        Assert.True(cmat.IsInitialized);
        Assert.Equal(Rectangle.Empty, cmat.Roi);
    }

    [Fact]
    public async Task Compress_MatRoiCalledConcurrently_NoDeadlockOrException()
    {
        using var sourceMat = CreateMat(64);
        var roiRect = new Rectangle(10, 10, 50, 40);

        var cmat = new CMat();
        var barrier = new Barrier(ThreadCount);

        var tasks = Enumerable.Range(0, ThreadCount).Select(_ => Task.Run(() =>
        {
            using var matRoi = new MatRoi(sourceMat, roiRect, leaveOpen: true);
            barrier.SignalAndWait(TestContext.Current.CancellationToken);
            cmat.Compress(matRoi);
        }, TestContext.Current.CancellationToken)).ToArray();

        await Task.WhenAll(tasks).WaitAsync(TestTimeout, TestContext.Current.CancellationToken);

        Assert.True(cmat.IsInitialized);
        Assert.Equal(100, cmat.Width);
        Assert.Equal(80, cmat.Height);
        Assert.Equal(roiRect, cmat.Roi);
    }

    // -------------------------------------------------------------------------
    // ChangeCompressor re-encode path (calls CompressInternal internally)
    // -------------------------------------------------------------------------

    [Fact]
    public async Task ChangeCompressor_ReEncodeWithNewCompressor_DoesNotDeadlock()
    {
        // reEncodeWithNewCompressor=true: RawDecompressInternal() + CompressInternal(mat) inside ChangeCompressor.
        using var mat = CreateMat(200);
        var cmat = new CMat(mat, MatCompressorBrotli.Instance);

        await Task.Run(() =>
            cmat.ChangeCompressor(MatCompressorDeflate.Instance, reEncodeWithNewCompressor: true),
            TestContext.Current.CancellationToken)
            .WaitAsync(TestTimeout, TestContext.Current.CancellationToken);

        Assert.Same(MatCompressorDeflate.Instance, cmat.Decompressor);
        Assert.True(cmat.IsInitialized);
    }

    [Fact]
    public async Task ChangeCompressorAsync_ConcurrentWithCompress_NoDeadlock()
    {
        using var mat = CreateMat(200);
        var cmat = new CMat(mat, MatCompressorBrotli.Instance);
        var barrier = new Barrier(2);

        var changeTask = Task.Run(() =>
        {
            barrier.SignalAndWait(TestContext.Current.CancellationToken);
            cmat.ChangeCompressor(MatCompressorDeflate.Instance, reEncodeWithNewCompressor: true);
        }, TestContext.Current.CancellationToken);

        var compressTask = Task.Run(() =>
        {
            using var m = CreateMat(100);
            barrier.SignalAndWait(TestContext.Current.CancellationToken);
            cmat.Compress(m);
        }, TestContext.Current.CancellationToken);

        await Task.WhenAll(changeTask, compressTask).WaitAsync(TestTimeout, TestContext.Current.CancellationToken);

        Assert.True(cmat.IsInitialized);
    }

    // -------------------------------------------------------------------------
    // Concurrent reads (multiple readers should not block each other)
    // -------------------------------------------------------------------------

    [Fact]
    public async Task Decompress_ConcurrentReads_AllSucceedSimultaneously()
    {
        const int pixelValue = 42;
        using var mat = CreateMat(pixelValue);
        var cmat = new CMat(mat);
        var expected = mat.ToArray();

        var barrier = new Barrier(ThreadCount);
        var results = new ConcurrentBag<byte[]>();

        var tasks = Enumerable.Range(0, ThreadCount).Select(_ => Task.Run(() =>
        {
            barrier.SignalAndWait(TestContext.Current.CancellationToken); // All readers start simultaneously
            using var decompressed = cmat.Decompress();
            results.Add(decompressed.ToArray());
        }, TestContext.Current.CancellationToken)).ToArray();

        await Task.WhenAll(tasks).WaitAsync(TestTimeout, TestContext.Current.CancellationToken);

        Assert.Equal(ThreadCount, results.Count);
        foreach (var result in results)
        {
            Assert.Equal(expected, result);
        }
    }

    [Fact]
    public async Task RawDecompress_ConcurrentReads_AllSucceed()
    {
        const int pixelValue = 99;
        using var mat = CreateMat(pixelValue);
        var cmat = new CMat(mat);
        var expected = mat.ToArray();

        var barrier = new Barrier(ThreadCount);
        var results = new ConcurrentBag<byte[]>();

        var tasks = Enumerable.Range(0, ThreadCount).Select(_ => Task.Run(() =>
        {
            barrier.SignalAndWait(TestContext.Current.CancellationToken);
            using var decompressed = cmat.RawDecompress();
            results.Add(decompressed.ToArray());
        }, TestContext.Current.CancellationToken)).ToArray();

        await Task.WhenAll(tasks).WaitAsync(TestTimeout, TestContext.Current.CancellationToken);

        Assert.Equal(ThreadCount, results.Count);
        foreach (var result in results)
        {
            Assert.Equal(expected, result);
        }
    }

    [Fact]
    public async Task Hash_ConcurrentReads_AllReturnSameValue()
    {
        using var mat = CreateMat(77);
        var cmat = new CMat(mat);

        var barrier = new Barrier(ThreadCount);
        var results = new ConcurrentBag<ulong>();

        var tasks = Enumerable.Range(0, ThreadCount).Select(_ => Task.Run(() =>
        {
            barrier.SignalAndWait(TestContext.Current.CancellationToken);
            results.Add(cmat.Hash);
        }, TestContext.Current.CancellationToken)).ToArray();

        await Task.WhenAll(tasks).WaitAsync(TestTimeout, TestContext.Current.CancellationToken);

        Assert.Equal(ThreadCount, results.Count);
        Assert.Single(results.Distinct()); // All should return the same hash
    }

    [Fact]
    public async Task CopyTo_ConcurrentReads_AllSucceed()
    {
        const int pixelValue = 55;
        using var mat = CreateMat(pixelValue);
        var cmat = new CMat(mat);

        var barrier = new Barrier(ThreadCount);
        var results = new ConcurrentBag<CMat>();

        var tasks = Enumerable.Range(0, ThreadCount).Select(_ => Task.Run(() =>
        {
            var dst = new CMat();
            barrier.SignalAndWait(TestContext.Current.CancellationToken);
            cmat.CopyTo(dst);
            results.Add(dst);
        }, TestContext.Current.CancellationToken)).ToArray();

        await Task.WhenAll(tasks).WaitAsync(TestTimeout, TestContext.Current.CancellationToken);

        Assert.Equal(ThreadCount, results.Count);
        foreach (var copy in results)
        {
            Assert.Equal(cmat.Width, copy.Width);
            Assert.Equal(cmat.Height, copy.Height);
            Assert.Equal(cmat.Hash, copy.Hash);
        }
    }

    [Fact]
    public async Task CopyTo_OppositeDirectionsConcurrent_NoDeadlock()
    {
        using var mat1 = CreateMat(55);
        using var mat2 = CreateMat(155);
        var cmat1 = new CMat(mat1);
        var cmat2 = new CMat(mat2);
        var barrier = new Barrier(2);

        var copy1 = Task.Run(() =>
        {
            barrier.SignalAndWait(TestContext.Current.CancellationToken);
            cmat1.CopyTo(cmat2);
        }, TestContext.Current.CancellationToken);

        var copy2 = Task.Run(() =>
        {
            barrier.SignalAndWait(TestContext.Current.CancellationToken);
            cmat2.CopyTo(cmat1);
        }, TestContext.Current.CancellationToken);

        await Task.WhenAll(copy1, copy2).WaitAsync(TestTimeout, TestContext.Current.CancellationToken);

        Assert.Equal(cmat1.Width, cmat2.Width);
        Assert.Equal(cmat1.Height, cmat2.Height);
    }

    // -------------------------------------------------------------------------
    // Concurrent reads + writes (readers wait for writers, writers wait for readers)
    // -------------------------------------------------------------------------

    [Fact]
    public async Task Decompress_ConcurrentWithCompress_NoExceptionOrCorruption()
    {
        // This scenario was previously out-of-scope. Now it is safe because
        // Decompress acquires a read lock and Compress acquires a write lock.
        const int pixelValue = 150;
        using var mat = CreateMat(pixelValue);
        var cmat = new CMat(mat);
        var errors = new ConcurrentBag<Exception>();

        using var cts = CancellationTokenSource.CreateLinkedTokenSource(TestContext.Current.CancellationToken);
        cts.CancelAfter(TestTimeout);

        // Writer task: repeatedly compresses new values
        var writerTask = Task.Run(() =>
        {
            for (int i = 0; i < 50 && !cts.IsCancellationRequested; i++)
            {
                try
                {
                    using var m = CreateMat((i * 5) % 256);
                    cmat.Compress(m);
                }
                catch (Exception ex)
                {
                    errors.Add(ex);
                }
            }
        }, TestContext.Current.CancellationToken);

        // Reader tasks: repeatedly decompress while writer is active
        var readerTasks = Enumerable.Range(0, ThreadCount).Select(_ => Task.Run(() =>
        {
            for (int i = 0; i < 50 && !cts.IsCancellationRequested; i++)
            {
                try
                {
                    using var decompressed = cmat.Decompress();
                    // Must always produce a valid mat with correct dimensions
                    Assert.Equal(100, decompressed.Width);
                    Assert.Equal(80, decompressed.Height);
                }
                catch (Exception ex) when (ex is not Xunit.Sdk.XunitException)
                {
                    errors.Add(ex);
                }
            }
        }, TestContext.Current.CancellationToken)).ToArray();

        await Task.WhenAll(readerTasks.Append(writerTask)).WaitAsync(TestTimeout, TestContext.Current.CancellationToken);

        Assert.Empty(errors);
    }

    // -------------------------------------------------------------------------
    // Stress test
    // -------------------------------------------------------------------------

    [Fact]
    public async Task Compress_HighConcurrencyStress_NoExceptionOrDeadlock()
    {
        const int iterations = 50;
        const int concurrency = 16;
        var cmat = new CMat();
        var errors = new ConcurrentBag<Exception>();

        var tasks = Enumerable.Range(0, concurrency).Select(i => Task.Run(() =>
        {
            for (int j = 0; j < iterations; j++)
            {
                try
                {
                    using var mat = CreateMat((i + j) % 256);
                    cmat.Compress(mat);
                }
                catch (Exception ex)
                {
                    errors.Add(ex);
                }
            }
        }, TestContext.Current.CancellationToken)).ToArray();

        await Task.WhenAll(tasks).WaitAsync(TimeSpan.FromSeconds(30), TestContext.Current.CancellationToken);

        Assert.Empty(errors);
        Assert.True(cmat.IsInitialized);
        Assert.Equal(100, cmat.Width);
        Assert.Equal(80, cmat.Height);
    }

    [Fact]
    public async Task MixedReadWrite_HighConcurrencyStress_NoExceptionOrDeadlock()
    {
        const int iterations = 50;
        const int writerCount = 4;
        const int readerCount = 12;

        using var mat = CreateMat(128);
        var cmat = new CMat(mat);
        var errors = new ConcurrentBag<Exception>();

        var writerTasks = Enumerable.Range(0, writerCount).Select(i => Task.Run(() =>
        {
            for (int j = 0; j < iterations; j++)
            {
                try
                {
                    using var m = CreateMat((i + j) % 256);
                    cmat.Compress(m);
                }
                catch (Exception ex)
                {
                    errors.Add(ex);
                }
            }
        }, TestContext.Current.CancellationToken)).ToArray();

        var readerTasks = Enumerable.Range(0, readerCount).Select(_ => Task.Run(() =>
        {
            for (int j = 0; j < iterations; j++)
            {
                try
                {
                    using var decompressed = cmat.Decompress();
                    Assert.Equal(100, decompressed.Width);
                    Assert.Equal(80, decompressed.Height);
                }
                catch (Exception ex) when (ex is not Xunit.Sdk.XunitException)
                {
                    errors.Add(ex);
                }
            }
        }, TestContext.Current.CancellationToken)).ToArray();

        await Task.WhenAll(writerTasks.Concat(readerTasks)).WaitAsync(TimeSpan.FromSeconds(30), TestContext.Current.CancellationToken);

        Assert.Empty(errors);
        Assert.True(cmat.IsInitialized);
    }
}
