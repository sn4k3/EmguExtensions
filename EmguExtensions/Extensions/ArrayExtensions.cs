
namespace EmguExtensions;

/// <summary>
/// Provides extension methods for arrays, including a method to create a copy of a byte array
/// without zeroing the memory for improved performance.
/// Use with caution as it may contain sensitive data.
/// </summary>
public static class ArrayExtensions
{
    extension(byte[] buffer)
    {
        /// <summary>
        /// Creates a copy of the byte array without zeroing the memory, which can be faster for large arrays. Use with caution as it may contain sensitive data.
        /// </summary>
        /// <returns></returns>
        public byte[] ToArrayPerf()
        {
            if (buffer.Length == 0) return [];
            var copy = GC.AllocateUninitializedArray<byte>(buffer.Length);
            buffer.CopyTo(copy);
            return copy;
        }
    }
}