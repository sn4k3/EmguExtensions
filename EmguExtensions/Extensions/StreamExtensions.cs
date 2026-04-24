namespace EmguExtensions
{
    /// <summary>
    /// Provides extension methods for <see cref="MemoryStream"/>, including a method to create a copy of the stream's buffer without zeroing the memory for improved performance.
    /// </summary>
    public static class StreamExtensions
    {
        extension(MemoryStream stream)
        {
            /// <summary>
            /// Returns the contents of the underlying stream as a byte array, using an optimized approach for
            /// performance.
            /// </summary>
            /// <remarks>This method may allocate an uninitialized array for performance when
            /// possible. The returned array is a copy of the stream's data and is not affected by subsequent changes to
            /// the stream.</remarks>
            /// <returns>A byte array containing the data from the stream. The array is empty if the stream has no data.</returns>
            public byte[] ToArrayPerf()
            {
                if (stream.Length == 0) return [];

                if (stream.TryGetBuffer(out var buffer))
                {
                    var copy = GC.AllocateUninitializedArray<byte>(buffer.Count);
                    buffer.AsSpan().CopyTo(copy);
                    return copy;
                }

                return stream.ToArray();
            }
        }
    }
}
