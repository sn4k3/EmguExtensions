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

#region

using System.Buffers;
using System.IO.Compression;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;
using Emgu.CV;

#endregion

namespace EmguExtensions;

/// <summary>
/// Wraps a stream compressor with a lossless delta filter: every element is replaced by its difference from the
/// previous element of the same channel in the row, and multi-byte elements are then split into byte planes.
/// </summary>
/// <remarks>
/// <para>The filter turns smooth, continuous-tone data (photos, depth maps, float fields) into small, repetitive values,
/// so the inner compressor shrinks it far more: e.g. Zstd goes from about 1.05x to 1.7-1.9x on such images, matching or
/// beating <see cref="MatCompressorTiff"/> while being several times faster. For masks and images with large flat areas
/// the filter does not help and only costs time: use the unfiltered compressor there.</para>
/// <para>Differences use wrapping integer arithmetic on the raw element bits (floats included), so any depth round trips
/// exactly. Rows are filtered independently, so ROI views and non-continuous matrices are supported. The encoded layout
/// is a storage format: it must stay stable across versions.</para>
/// </remarks>
public sealed class MatCompressorDelta : MatCompressor
{
    /// <summary>
    /// <see cref="MatCompressorDeflate"/> with the delta filter.
    /// </summary>
    public static readonly MatCompressorDelta Deflate = new(MatCompressorDeflate.Instance);

    /// <summary>
    /// <see cref="MatCompressorBrotli"/> with the delta filter.
    /// </summary>
    public static readonly MatCompressorDelta Brotli = new(MatCompressorBrotli.Instance);

#if NET11_0_OR_GREATER
    /// <summary>
    /// <see cref="MatCompressorZstd"/> with the delta filter. The fastest option with the best ratio on continuous-tone
    /// data.
    /// </summary>
    public static readonly MatCompressorDelta Zstd = new(MatCompressorZstd.Instance);
#endif

    /// <summary>
    /// Creates a delta-filtered compressor around <paramref name="inner"/>. Register it in
    /// <see cref="MatCompressor.AvailableCompressors"/> if <see cref="CMat"/> must find it by id.
    /// </summary>
    /// <param name="inner">The byte compressor that stores the filtered data.</param>
    /// <exception cref="ArgumentException"><paramref name="inner"/> is already a <see cref="MatCompressorDelta"/> or an
    /// image codec that allocates its own destination (e.g. PNG, TIFF), which cannot store filtered bytes.</exception>
    public MatCompressorDelta(MatCompressor inner)
    {
        ArgumentNullException.ThrowIfNull(inner);
        if (inner is MatCompressorDelta || AllocatesDestinationOf(inner))
        {
            throw new ArgumentException(
                $"The compressor '{inner.Name}' cannot be wrapped: it must be a byte compressor.",
                nameof(inner)
            );
        }

        Inner = inner;
        Name = $"{inner.Name}+Delta";
    }

    /// <summary>
    /// Gets the compressor that stores the filtered data.
    /// </summary>
    public MatCompressor Inner { get; }

    /// <inheritdoc />
    public override string Provider => Inner.Provider;

    /// <inheritdoc />
    public override string Name { get; }

    /// <inheritdoc />
    public override int MinimumCompressionLevel => Inner.MinimumCompressionLevel;

    /// <inheritdoc />
    public override int MaximumCompressionLevel => Inner.MaximumCompressionLevel;

    /// <inheritdoc />
    protected override int GetCompressionLevel(CompressionLevel compressionLevel)
    {
        return GetCompressionLevelOf(Inner, compressionLevel);
    }

    /// <inheritdoc />
    protected override byte[] CompressCore(Mat src, int compressionLevel)
    {
        var elementSize = src.Depth.ByteCount;
        var channels = src.NumberOfChannels;
        var rowLength = src.Width * src.ElementSize;
        using var filtered = new Mat(src.Size, src.Depth, channels);
        var planes = elementSize > 1 ? ArrayPool<byte>.Shared.Rent(rowLength) : null;
        try
        {
            for (var y = 0; y < src.Height; y++)
            {
                var input = src.GetReadOnlyRowSpanOfBytes(y);
                var output = filtered.GetRowSpanOfBytes(y);
                if (planes is null)
                {
                    DeltaEncode(input, output, channels);
                }
                else
                {
                    var deltas = planes.AsSpan(0, rowLength);
                    DeltaEncodeRow(input, deltas, elementSize, channels);
                    Shuffle(deltas, output, elementSize);
                }
            }
        }
        finally
        {
            if (planes is not null)
            {
                ArrayPool<byte>.Shared.Return(planes);
            }
        }

        return Inner.Compress(filtered, compressionLevel);
    }

    /// <inheritdoc />
    protected override void DecompressCore(byte[] compressedBytes, Mat dst)
    {
        var elementSize = dst.Depth.ByteCount;
        var channels = dst.NumberOfChannels;

        // The filtered rows have the same shape as dst, so the inner compressor decodes straight into it and the
        // filter is undone in place (one pooled row buffer for the byte planes, no image-sized temporary)
        Inner.Decompress(compressedBytes, dst);

        var rowLength = dst.Width * dst.ElementSize;
        var planes = elementSize > 1 ? ArrayPool<byte>.Shared.Rent(rowLength) : null;
        try
        {
            for (var y = 0; y < dst.Height; y++)
            {
                var row = dst.GetRowSpanOfBytes(y);
                if (planes is not null)
                {
                    var rowPlanes = planes.AsSpan(0, rowLength);
                    row.CopyTo(rowPlanes);
                    Unshuffle(rowPlanes, row, elementSize);
                }

                DeltaDecodeRow(row, elementSize, channels);
            }
        }
        finally
        {
            if (planes is not null)
            {
                ArrayPool<byte>.Shared.Return(planes);
            }
        }
    }

    #region Filter

    // 128-bit byte shuffle (out-of-range indices, e.g. 0xFF, produce zero): SSSE3 pshufb or ARM64 tbl.
    private static bool CanShuffleBytes => Ssse3.IsSupported || AdvSimd.Arm64.IsSupported;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<byte> ShuffleBytes(Vector128<byte> vector, Vector128<byte> indices)
    {
        return Ssse3.IsSupported
            ? Ssse3.Shuffle(vector, indices)
            : AdvSimd.Arm64.VectorTableLookup(vector, indices);
    }

    private static void DeltaEncodeRow(
        ReadOnlySpan<byte> src,
        Span<byte> dst,
        int elementSize,
        int channels
    )
    {
        switch (elementSize)
        {
            case 2:
                DeltaEncode(
                    MemoryMarshal.Cast<byte, ushort>(src),
                    MemoryMarshal.Cast<byte, ushort>(dst),
                    channels
                );
                break;
            case 4:
                DeltaEncode(
                    MemoryMarshal.Cast<byte, uint>(src),
                    MemoryMarshal.Cast<byte, uint>(dst),
                    channels
                );
                break;
            case 8:
                DeltaEncode(
                    MemoryMarshal.Cast<byte, ulong>(src),
                    MemoryMarshal.Cast<byte, ulong>(dst),
                    channels
                );
                break;
            default:
                DeltaEncode(src, dst, channels);
                break;
        }
    }

    private static void DeltaDecodeRow(Span<byte> row, int elementSize, int channels)
    {
        switch (elementSize)
        {
            case 2:
                DeltaDecode<ushort>(row, channels);
                break;
            case 4:
                DeltaDecode<uint>(row, channels);
                break;
            case 8:
                DeltaDecode<ulong>(row, channels);
                break;
            default:
                DeltaDecode<byte>(row, channels);
                break;
        }
    }

    // dst[i] = src[i] - src[i - channels], wrapping. Independent per element, so it vectorizes.
    private static void DeltaEncode<T>(ReadOnlySpan<T> src, Span<T> dst, int channels)
        where T : unmanaged, IBinaryInteger<T>
    {
        var head = Math.Min(channels, src.Length);
        src[..head].CopyTo(dst);
        var i = head;
        if (Vector.IsHardwareAccelerated && Vector<T>.IsSupported)
        {
            for (; i <= src.Length - Vector<T>.Count; i += Vector<T>.Count)
            {
                var current = new Vector<T>(src[i..]);
                var previous = new Vector<T>(src[(i - channels)..]);
                (current - previous).CopyTo(dst[i..]);
            }
        }

        for (; i < src.Length; i++)
        {
            dst[i] = unchecked(src[i] - src[i - channels]);
        }
    }

    // Running sum per channel, wrapping, in place: the exact inverse of DeltaEncode.
    private static void DeltaDecode<T>(Span<byte> row, int channels)
        where T : unmanaged, IBinaryInteger<T>
    {
        const int blockSize = 16; // Vector128<byte>.Count
        var elementSize = Unsafe.SizeOf<T>();
        var period = channels * elementSize; // Bytes between consecutive elements of the same channel
        var i = 0; // Bytes before i are final
        ref var start = ref MemoryMarshal.GetReference(row);

        if (Vector128.IsHardwareAccelerated && row.Length >= blockSize)
        {
            if (period >= blockSize)
            {
                // The previous element of every lane is in an earlier, already final block: a plain vector add
                for (i = period; i <= row.Length - blockSize; i += blockSize)
                {
                    var current = Vector128.LoadUnsafe(ref start, (nuint)i).As<byte, T>();
                    var previous = Vector128
                        .LoadUnsafe(ref start, (nuint)(i - period))
                        .As<byte, T>();
                    (current + previous).AsByte().StoreUnsafe(ref start, (nuint)i);
                }
            }
            else if (CanShuffleBytes)
            {
                // Log-step scan inside each block (shift by period, 2*period, ... bytes and add), then add the
                // running total of each channel carried from the previous block
                Span<Vector128<byte>> steps = stackalloc Vector128<byte>[4];
                var stepCount = 0;
                for (var shift = period; shift < blockSize; shift *= 2)
                {
                    steps[stepCount++] = CreateShiftMask(shift);
                }

                var carryMask = CreateCarryMask(period);
                var carry = Vector128<T>.Zero;
                for (; i <= row.Length - blockSize; i += blockSize)
                {
                    var block = Vector128.LoadUnsafe(ref start, (nuint)i).As<byte, T>();
                    for (var s = 0; s < stepCount; s++)
                    {
                        block += ShuffleBytes(block.AsByte(), steps[s]).As<byte, T>();
                    }

                    block += ShuffleBytes(carry.AsByte(), carryMask).As<byte, T>();
                    block.AsByte().StoreUnsafe(ref start, (nuint)i);
                    carry = block;
                }
            }
        }

        // Scalar for the tail (or everything without SIMD)
        var elements = MemoryMarshal.Cast<byte, T>(row);
        for (var e = Math.Max(i / elementSize, channels); e < elements.Length; e++)
        {
            elements[e] = unchecked(elements[e] + elements[e - channels]);
        }
    }

    // Lane j takes byte j - shift (zero below shift).
    private static Vector128<byte> CreateShiftMask(int shift)
    {
        Span<byte> mask = stackalloc byte[16];
        for (var j = 0; j < mask.Length; j++)
        {
            mask[j] = j >= shift ? (byte)(j - shift) : byte.MaxValue;
        }

        return Vector128.Create(mask);
    }

    // Lane j takes, from the previous block, the last byte of the same channel and byte position.
    private static Vector128<byte> CreateCarryMask(int period)
    {
        Span<byte> mask = stackalloc byte[16];
        for (var j = 0; j < mask.Length; j++)
        {
            mask[j] = (byte)(mask.Length + j - period * (j / period + 1));
        }

        return Vector128.Create(mask);
    }

    // Shuffled block lane j = plane (j / group), element (j % group): source byte element * elementSize + plane.
    private static Vector128<byte> CreatePlaneMask(int elementSize)
    {
        Span<byte> mask = stackalloc byte[16];
        var group = mask.Length / elementSize;
        for (var j = 0; j < mask.Length; j++)
        {
            mask[j] = (byte)(j % group * elementSize + j / group);
        }

        return Vector128.Create(mask);
    }

    // Inverse of CreatePlaneMask: lane element * elementSize + plane takes shuffled byte plane * group + element.
    private static Vector128<byte> CreateInversePlaneMask(int elementSize)
    {
        Span<byte> mask = stackalloc byte[16];
        var group = mask.Length / elementSize;
        for (var j = 0; j < mask.Length; j++)
        {
            mask[j] = (byte)(j % elementSize * group + j / elementSize);
        }

        return Vector128.Create(mask);
    }

    // Byte planes: byte 0 of every element, then byte 1, ... (high bytes of small deltas become long zero runs).
    private static void Shuffle(ReadOnlySpan<byte> src, Span<byte> dst, int elementSize)
    {
        var count = src.Length / elementSize;
        var i = 0; // Bytes of src already processed
        if (Vector128.IsHardwareAccelerated && CanShuffleBytes && elementSize is 2 or 4 or 8)
        {
            ref var source = ref MemoryMarshal.GetReference(src);
            ref var destination = ref MemoryMarshal.GetReference(dst);
            var mask = CreatePlaneMask(elementSize);
            for (; i <= src.Length - 16; i += 16)
            {
                // One block holds 16 / elementSize elements: write each plane's group at its offset
                var block = ShuffleBytes(Vector128.LoadUnsafe(ref source, (nuint)i), mask);
                var element = i / elementSize;
                switch (elementSize)
                {
                    case 2:
                        var groups64 = block.AsUInt64();
                        for (var b = 0; b < 2; b++)
                        {
                            Unsafe.WriteUnaligned(
                                ref Unsafe.Add(ref destination, b * count + element),
                                groups64.GetElement(b)
                            );
                        }
                        break;
                    case 4:
                        var groups32 = block.AsUInt32();
                        for (var b = 0; b < 4; b++)
                        {
                            Unsafe.WriteUnaligned(
                                ref Unsafe.Add(ref destination, b * count + element),
                                groups32.GetElement(b)
                            );
                        }
                        break;
                    default:
                        var groups16 = block.AsUInt16();
                        for (var b = 0; b < 8; b++)
                        {
                            Unsafe.WriteUnaligned(
                                ref Unsafe.Add(ref destination, b * count + element),
                                groups16.GetElement(b)
                            );
                        }
                        break;
                }
            }
        }

        for (var b = 0; b < elementSize; b++)
        {
            var plane = dst.Slice(b * count, count);
            for (int k = i / elementSize, j = i + b; k < plane.Length; k++, j += elementSize)
            {
                plane[k] = src[j];
            }
        }
    }

    private static void Unshuffle(ReadOnlySpan<byte> src, Span<byte> dst, int elementSize)
    {
        var count = src.Length / elementSize;
        var i = 0; // Bytes of dst already written
        if (Vector128.IsHardwareAccelerated && CanShuffleBytes && elementSize is 2 or 4 or 8)
        {
            ref var source = ref MemoryMarshal.GetReference(src);
            ref var destination = ref MemoryMarshal.GetReference(dst);
            var mask = CreateInversePlaneMask(elementSize);
            for (; i <= dst.Length - 16; i += 16)
            {
                // Gather each plane's group for these 16 / elementSize elements, then interleave them back
                var element = i / elementSize;
                var block = elementSize switch
                {
                    2 => Vector128
                        .Create(
                            Unsafe.ReadUnaligned<ulong>(ref Unsafe.Add(ref source, element)),
                            Unsafe.ReadUnaligned<ulong>(ref Unsafe.Add(ref source, count + element))
                        )
                        .AsByte(),
                    4 => Vector128
                        .Create(
                            Unsafe.ReadUnaligned<uint>(ref Unsafe.Add(ref source, element)),
                            Unsafe.ReadUnaligned<uint>(ref Unsafe.Add(ref source, count + element)),
                            Unsafe.ReadUnaligned<uint>(
                                ref Unsafe.Add(ref source, 2 * count + element)
                            ),
                            Unsafe.ReadUnaligned<uint>(
                                ref Unsafe.Add(ref source, 3 * count + element)
                            )
                        )
                        .AsByte(),
                    _ => Vector128
                        .Create(
                            Unsafe.ReadUnaligned<ushort>(ref Unsafe.Add(ref source, element)),
                            Unsafe.ReadUnaligned<ushort>(
                                ref Unsafe.Add(ref source, count + element)
                            ),
                            Unsafe.ReadUnaligned<ushort>(
                                ref Unsafe.Add(ref source, 2 * count + element)
                            ),
                            Unsafe.ReadUnaligned<ushort>(
                                ref Unsafe.Add(ref source, 3 * count + element)
                            ),
                            Unsafe.ReadUnaligned<ushort>(
                                ref Unsafe.Add(ref source, 4 * count + element)
                            ),
                            Unsafe.ReadUnaligned<ushort>(
                                ref Unsafe.Add(ref source, 5 * count + element)
                            ),
                            Unsafe.ReadUnaligned<ushort>(
                                ref Unsafe.Add(ref source, 6 * count + element)
                            ),
                            Unsafe.ReadUnaligned<ushort>(
                                ref Unsafe.Add(ref source, 7 * count + element)
                            )
                        )
                        .AsByte(),
                };
                ShuffleBytes(block, mask).StoreUnsafe(ref destination, (nuint)i);
            }
        }

        for (var b = 0; b < elementSize; b++)
        {
            var plane = src.Slice(b * count, count);
            for (int k = i / elementSize, j = i + b; k < plane.Length; k++, j += elementSize)
            {
                dst[j] = plane[k];
            }
        }
    }

    #endregion
}
