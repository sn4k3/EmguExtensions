#pragma warning disable xUnit1051

using System.Runtime.InteropServices;
using Avalonia;
using Avalonia.Headless;
using Avalonia.Media.Imaging;
using Avalonia.Platform;
using Avalonia.Skia;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using EmguExtensions.Avalonia;
using Xunit;
using DrawingPoint = System.Drawing.Point;
using DrawingRectangle = System.Drawing.Rectangle;
using DrawingSize = System.Drawing.Size;

namespace EmguExtensions.Tests;

/// <summary>
/// Tests for the EmguExtensions.Avalonia project, running on the Avalonia headless platform.
/// </summary>
public class UnitTestAvalonia
{
    private sealed class TestApplication : Application;

    private static readonly Lazy<bool> Platform = new(() =>
    {
        AppBuilder
            .Configure<TestApplication>()
            .UseSkia()
            .UseHeadless(new AvaloniaHeadlessPlatformOptions { UseHeadlessDrawing = false })
            .SetupWithoutStarting();
        return true;
    });

    public UnitTestAvalonia()
    {
        _ = Platform.Value;
    }

    /// <summary>
    /// A framebuffer over pinned managed memory with a configurable stride and format.
    /// </summary>
    private sealed class FakeFramebuffer : ILockedFramebuffer
    {
        private readonly byte[] _buffer;
        private GCHandle _handle;

        public FakeFramebuffer(int width, int height, int rowBytes, PixelFormat? format = null)
        {
            _buffer = new byte[rowBytes * height];
            _handle = GCHandle.Alloc(_buffer, GCHandleType.Pinned);
            Size = new PixelSize(width, height);
            RowBytes = rowBytes;
            Format = format ?? PixelFormat.Bgra8888;
        }

        public byte[] Buffer => _buffer;
        public IntPtr Address => _handle.AddrOfPinnedObject();
        public PixelSize Size { get; }
        public int RowBytes { get; }
        public Vector Dpi => new(96, 96);
        public PixelFormat Format { get; }
        public AlphaFormat AlphaFormat => AlphaFormat.Unpremul;

        public void Dispose()
        {
            if (_handle.IsAllocated)
                _handle.Free();
        }
    }

    private static uint[] ReadPixels(WriteableBitmap bitmap)
    {
        using var framebuffer = bitmap.Lock();
        var pixels = new uint[bitmap.PixelSize.Width * bitmap.PixelSize.Height];
        framebuffer.GetReadOnlySpan2D().CopyTo(pixels);
        return pixels;
    }

    private static uint Bgra(byte b, byte g, byte r, byte a) => (uint)(a << 24 | r << 16 | g << 8 | b);

    private static Mat CreateColorMat(int width, int height, int channels, MCvScalar color)
    {
        var mat = new Mat(new DrawingSize(width, height), DepthType.Cv8U, channels);
        mat.SetTo(color);
        return mat;
    }

    #region ToBitmap

    [Fact]
    public void ToBitmap_Gray_ExpandsToOpaqueBgra()
    {
        using var mat = CreateColorMat(8, 6, 1, new MCvScalar(100));

        using var bitmap = mat.ToBitmap();

        Assert.Equal(new PixelSize(8, 6), bitmap.PixelSize);
        Assert.Equal(PixelFormat.Bgra8888, bitmap.Format);
        Assert.All(ReadPixels(bitmap), p => Assert.Equal(Bgra(100, 100, 100, 255), p));
    }

    [Fact]
    public void ToBitmap_Bgr_KeepsChannelOrderAndAddsOpaqueAlpha()
    {
        using var mat = CreateColorMat(5, 4, 3, new MCvScalar(10, 20, 30));

        using var bitmap = mat.ToBitmap();

        Assert.All(ReadPixels(bitmap), p => Assert.Equal(Bgra(10, 20, 30, 255), p));
    }

    [Fact]
    public void ToBitmap_Bgra_KeepsAlpha()
    {
        using var mat = CreateColorMat(5, 4, 4, new MCvScalar(10, 20, 30, 128));

        using var bitmap = mat.ToBitmap();

        Assert.All(ReadPixels(bitmap), p => Assert.Equal(Bgra(10, 20, 30, 128), p));
    }

    [Fact]
    public void ToBitmap_SrcType_Rgba_SwapsRedAndBlue()
    {
        // Mat in RGBA order: R=200, G=100, B=50, A=128
        using var mat = CreateColorMat(4, 4, 4, new MCvScalar(200, 100, 50, 128));

        using var bitmap = mat.ToBitmap(typeof(Rgba));

        Assert.All(ReadPixels(bitmap), p => Assert.Equal(Bgra(50, 100, 200, 128), p));
    }

    [Fact]
    public void ToBitmap_SrcType_Hsv_Converts()
    {
        // H=0, S=255, V=255 is pure red
        using var mat = CreateColorMat(4, 4, 3, new MCvScalar(0, 255, 255));

        using var bitmap = mat.ToBitmap(typeof(Hsv));

        Assert.All(ReadPixels(bitmap), p => Assert.Equal(Bgra(0, 0, 255, 255), p));
    }

    [Fact]
    public void ToBitmap_SrcType_MatchingDefaults_Work()
    {
        using var gray = CreateColorMat(4, 4, 1, new MCvScalar(7));
        using var bgr = CreateColorMat(4, 4, 3, new MCvScalar(1, 2, 3));
        using var bgra = CreateColorMat(4, 4, 4, new MCvScalar(1, 2, 3, 4));

        using var a = gray.ToBitmap(typeof(Gray));
        using var b = bgr.ToBitmap(typeof(Bgr));
        using var c = bgra.ToBitmap(typeof(Bgra));

        Assert.All(ReadPixels(a), p => Assert.Equal(Bgra(7, 7, 7, 255), p));
        Assert.All(ReadPixels(b), p => Assert.Equal(Bgra(1, 2, 3, 255), p));
        Assert.All(ReadPixels(c), p => Assert.Equal(Bgra(1, 2, 3, 4), p));
    }

    [Fact]
    public void ToBitmap_SrcType_ChannelMismatch_Throws()
    {
        using var mat = CreateColorMat(4, 4, 3, new MCvScalar(1, 2, 3));

        Assert.Throws<ArgumentException>(() => mat.ToBitmap(typeof(Rgba)));
        Assert.Throws<ArgumentException>(() => mat.ToBitmap(typeof(Gray)));
    }

    [Fact]
    public void ToBitmap_SrcType_NotAColor_Throws()
    {
        using var mat = CreateColorMat(4, 4, 3, new MCvScalar(1, 2, 3));

        Assert.Throws<ArgumentException>(() => mat.ToBitmap(typeof(string)));
        Assert.Throws<ArgumentNullException>(() => mat.ToBitmap((Type)null!));
    }

    [Fact]
    public void ToBitmap_EmptyMat_Throws()
    {
        using var mat = new Mat();

        Assert.Throws<ArgumentException>(() => mat.ToBitmap());
    }

    [Fact]
    public void ToBitmap_UnsupportedChannelCount_ThrowsBeforeAllocating()
    {
        using var mat = CreateColorMat(4, 4, 2, new MCvScalar(1, 2));

        Assert.Throws<NotSupportedException>(() => mat.ToBitmap());
    }

    [Fact]
    public void ToBitmap_NonEightBit_ThrowsAndMentionsScale()
    {
        using var mat = new Mat(new DrawingSize(4, 4), DepthType.Cv16U, 1);

        var ex = Assert.Throws<NotSupportedException>(() => mat.ToBitmap());

        Assert.Contains("scale", ex.Message);
    }

    [Fact]
    public void ToBitmap_Scaled_16Bit_MapsFullRangeTo8Bit()
    {
        using var mat = new Mat(new DrawingSize(4, 4), DepthType.Cv16U, 1);
        mat.SetTo(new MCvScalar(65535));

        using var bitmap = mat.ToBitmap(255.0 / 65535);

        Assert.All(ReadPixels(bitmap), p => Assert.Equal(Bgra(255, 255, 255, 255), p));
    }

    [Fact]
    public void ToBitmap_Scaled_Float_UsesScaleAndShift()
    {
        using var mat = new Mat(new DrawingSize(4, 4), DepthType.Cv32F, 1);
        mat.SetTo(new MCvScalar(0.5));

        using var bitmap = mat.ToBitmap(200, 28); // 0.5 * 200 + 28 = 128

        Assert.All(ReadPixels(bitmap), p => Assert.Equal(Bgra(128, 128, 128, 255), p));
    }

    [Fact]
    public void ToBitmap_Scaled_InvalidFactor_Throws()
    {
        using var mat = CreateColorMat(4, 4, 1, new MCvScalar(1));

        Assert.Throws<ArgumentOutOfRangeException>(() => mat.ToBitmap(double.NaN));
    }

    [Fact]
    public void ToBitmap_NonContinuousRoi_MatchesItsClone()
    {
        using var big = new Mat(new DrawingSize(40, 30), DepthType.Cv8U, 3);
        var span = big.GetSpan<byte>();
        for (var i = 0; i < span.Length; i++)
            span[i] = (byte)(i * 7);

        using var roi = new Mat(big, new DrawingRectangle(5, 4, 17, 11));
        using var clone = roi.Clone();
        Assert.False(roi.IsContinuous);

        using var fromRoi = roi.ToBitmap();
        using var fromClone = clone.ToBitmap();

        Assert.Equal(ReadPixels(fromClone), ReadPixels(fromRoi));
    }

    [Fact]
    public void ToBitmap_Dpi_DefaultsTo96AndHonorsCustomValue()
    {
        using var mat = CreateColorMat(4, 4, 1, new MCvScalar(1));

        using var defaultDpi = mat.ToBitmap();
        using var custom = mat.ToBitmap(new Vector(300, 300));

        Assert.Equal(new Vector(96, 96), defaultDpi.Dpi);
        Assert.Equal(new Vector(300, 300), custom.Dpi);
    }

    [Fact]
    public async Task ToBitmapAsync_Overloads_ProduceTheSameBitmap()
    {
        using var mat = CreateColorMat(6, 5, 3, new MCvScalar(9, 8, 7));

        using var plain = await mat.ToBitmapAsync();
        using var typed = await mat.ToBitmapAsync(typeof(Bgr));
        using var scaled = await mat.ToBitmapAsync(1.0);

        var expected = Bgra(9, 8, 7, 255);
        Assert.All(ReadPixels(plain), p => Assert.Equal(expected, p));
        Assert.All(ReadPixels(typed), p => Assert.Equal(expected, p));
        Assert.All(ReadPixels(scaled), p => Assert.Equal(expected, p));
    }

    #endregion

    #region WriteableBitmap

    [Fact]
    public void WriteableBitmap_ToMat_RoundTripsThePixels()
    {
        using var source = new Mat(new DrawingSize(9, 7), DepthType.Cv8U, 4);
        var span = source.GetSpan<byte>();
        for (var i = 0; i < span.Length; i++)
            span[i] = (byte)(i * 13 + 5);
        using var bitmap = source.ToBitmap();

        using var back = bitmap.ToMat();

        Assert.Equal(4, back.NumberOfChannels);
        Assert.True(back.IsContinuous);
        Assert.Equal(source.ToArray(), back.ToArray());
    }

    [Fact]
    public void WriteableBitmap_GetBitmapInfo_DoesNotLeakTheAddress()
    {
        using var mat = CreateColorMat(10, 3, 3, new MCvScalar(1, 2, 3));
        using var bitmap = mat.ToBitmap();

        var info = bitmap.GetBitmapInfo();

        Assert.Equal(IntPtr.Zero, info.Address);
        Assert.Equal(10, info.Width);
        Assert.Equal(3, info.Height);
        Assert.Equal(4, info.BytesPerPixel);
        Assert.True(info.RowBytes >= 40);
    }

    [Fact]
    public void WriteableBitmap_WithBitmapInfo_ExposesTheAddressInsideTheCallback()
    {
        using var mat = CreateColorMat(10, 3, 3, new MCvScalar(1, 2, 3));
        using var bitmap = mat.ToBitmap();

        var firstPixel = bitmap.WithBitmapInfo(info =>
        {
            Assert.NotEqual(IntPtr.Zero, info.Address);
            return Marshal.ReadInt32(info.Address);
        });
        var calls = 0;
        bitmap.WithBitmapInfo(_ => calls++);

        Assert.Equal((int)Bgra(1, 2, 3, 255), firstPixel);
        Assert.Equal(1, calls);
        Assert.Throws<ArgumentNullException>(() => bitmap.WithBitmapInfo((Action<BitmapInfo>)null!));
    }

    #endregion

    #region ILockedFramebuffer

    [Fact]
    public void Framebuffer_ToMat_IsAZeroCopyViewHonoringTheStride()
    {
        using var framebuffer = new FakeFramebuffer(5, 3, 32); // 20 data bytes + 12 padding per row

        using (var view = framebuffer.ToMat())
        {
            Assert.Equal(new DrawingSize(5, 3), view.Size);
            Assert.Equal(4, view.NumberOfChannels);
            Assert.False(view.IsContinuous);
            view.SetTo(new MCvScalar(1, 2, 3, 4));
        }

        // Written through the view into the framebuffer memory, without touching the padding
        Assert.Equal([1, 2, 3, 4], framebuffer.Buffer[..4]);
        Assert.Equal([1, 2, 3, 4], framebuffer.Buffer[16..20]);
        Assert.All(framebuffer.Buffer[20..32], b => Assert.Equal(0, b));
        Assert.Equal([1, 2, 3, 4], framebuffer.Buffer[32..36]);
    }

    [Fact]
    public void Framebuffer_ToMat_UnsupportedFormat_Throws()
    {
        using var framebuffer = new FakeFramebuffer(4, 4, 8, PixelFormat.Rgb565);

        Assert.Throws<NotSupportedException>(() => framebuffer.ToMat());
    }

    [Fact]
    public void Framebuffer_Properties_AreComputedFromTheStride()
    {
        using var padded = new FakeFramebuffer(5, 3, 32);
        using var tight = new FakeFramebuffer(5, 3, 20);

        Assert.Equal(4, padded.BytesPerPixel);
        Assert.Equal(96, padded.ByteCount);
        Assert.Equal(96, padded.ByteCountInt64);
        Assert.Equal(15, padded.PixelCount);
        Assert.False(padded.IsContinuous);
        Assert.True(tight.IsContinuous);
    }

    [Fact]
    public void Framebuffer_GetBitmapInfo_ReportsTheRealStride()
    {
        using var padded = new FakeFramebuffer(5, 3, 32);

        var info = padded.GetBitmapInfo();

        Assert.Equal(32, info.RowBytes);
        Assert.False(info.IsContiguous);
        Assert.Equal(padded.Address, info.Address);
    }

    [Fact]
    public void Framebuffer_RowSpanOfBytes_ExcludesPaddingByDefault()
    {
        using var framebuffer = new FakeFramebuffer(5, 3, 32);

        Assert.Equal(20, framebuffer.GetRowSpanOfBytes(1).Length);
        Assert.Equal(16, framebuffer.GetRowSpanOfBytes(1, offset: 4).Length);
        Assert.Equal(32, framebuffer.GetRowSpanOfBytes(1, length: 32).Length); // Explicit length can reach the padding
        Assert.Throws<ArgumentOutOfRangeException>(() => framebuffer.GetRowSpanOfBytes(1, length: 33));
        Assert.Throws<ArgumentOutOfRangeException>(() => framebuffer.GetRowSpanOfBytes(1, offset: 21));
        Assert.Throws<ArgumentOutOfRangeException>(() => framebuffer.GetRowSpanOfBytes(3));
        Assert.Throws<ArgumentOutOfRangeException>(() => framebuffer.GetRowSpanOfBytes(0, length: -1));
    }

    [Fact]
    public void Framebuffer_RowSpan_AddressesTheRightRow()
    {
        using var framebuffer = new FakeFramebuffer(5, 3, 32);
        framebuffer.GetRowSpan(2)[1] = 0xAABBCCDD;

        Assert.Equal(0xDD, framebuffer.Buffer[2 * 32 + 4]);
        Assert.Equal(5, framebuffer.GetRowSpan(2).Length);
        Assert.Equal(3, framebuffer.GetRowSpan(2, offset: 2).Length);
        Assert.Throws<ArgumentOutOfRangeException>(() => framebuffer.GetRowSpan(2, length: 6));
    }

    [Fact]
    public void Framebuffer_FlatSpans_RequireContinuousAndRejectNegativeLength()
    {
        using var padded = new FakeFramebuffer(5, 3, 32);
        using var tight = new FakeFramebuffer(5, 3, 20);

        Assert.Throws<NotSupportedException>(() => padded.GetSpan());
        Assert.Equal(15, tight.GetSpan().Length);
        Assert.Equal(10, tight.GetSpan(offset: 5).Length);
        Assert.Throws<ArgumentOutOfRangeException>(() => tight.GetSpan(length: -1));
        Assert.Equal(60, tight.GetSpanOfBytes().Length);
        Assert.Equal(96, padded.GetSpanOfBytes().Length);
        Assert.Throws<ArgumentOutOfRangeException>(() => tight.GetSpanOfBytes(length: -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => tight.GetSpanOfBytes(offset: 61));
    }

    [Fact]
    public void Framebuffer_PixelPositions_AreValidated()
    {
        using var framebuffer = new FakeFramebuffer(5, 3, 32);

        Assert.Equal(32 * 2 + 4 * 3, framebuffer.GetPixelBytePos(3, 2));
        Assert.Equal(32 * 2 + 4 * 3, framebuffer.GetPixelBytePos(new DrawingPoint(3, 2)));
        Assert.Equal(32 * 2 + 4 * 3, framebuffer.GetPixelBytePos(new PixelPoint(3, 2)));
        Assert.Equal(5 * 2 + 3, framebuffer.GetPixelPos(3, 2));
        Assert.Equal(5 * 2 + 3, framebuffer.GetPixelPos(new DrawingPoint(3, 2)));
        Assert.Equal(5 * 2 + 3, framebuffer.GetPixelPos(new PixelPoint(3, 2)));
        Assert.Throws<ArgumentOutOfRangeException>(() => framebuffer.GetPixelBytePos(5, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => framebuffer.GetPixelBytePos(0, 3));
        Assert.Throws<ArgumentOutOfRangeException>(() => framebuffer.GetPixelPos(-1, 0));
    }

    [Fact]
    public void Framebuffer_Span2D_ExcludesPaddingAndSupportsPixelRect()
    {
        using var framebuffer = new FakeFramebuffer(5, 4, 32);
        var all = framebuffer.GetSpan2D();
        for (var y = 0; y < all.Height; y++)
            for (var x = 0; x < all.Width; x++)
                all[y, x] = (uint)(y * 10 + x);

        var roi = framebuffer.GetSpan2D(new PixelRect(1, 1, 3, 2));
        var roiFromDrawing = framebuffer.GetReadOnlySpan2D(new DrawingRectangle(1, 1, 3, 2));
        var bytes = framebuffer.GetSpan2DOfBytes(new PixelRect(1, 1, 3, 2));

        Assert.Equal(new uint[] { 11, 12, 13, 21, 22, 23 }, roi.ToArray().Cast<uint>());
        Assert.Equal(roi.ToArray().Cast<uint>(), roiFromDrawing.ToArray().Cast<uint>());
        Assert.Equal((3 * 4, 2), (bytes.Width, bytes.Height));
        Assert.Equal(20, framebuffer.GetSpan2DOfBytes().Width);
        Assert.Throws<ArgumentOutOfRangeException>(() => framebuffer.GetSpan2D(new PixelRect(3, 0, 3, 1)));
        Assert.Throws<ArgumentOutOfRangeException>(() => framebuffer.GetSpan2DOfBytes(new DrawingRectangle(-1, 0, 1, 1)));
    }

    #endregion
}
