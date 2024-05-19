using Cupy;
using Numpy;
using Python.Runtime;


namespace DeZero.NET
{
    public static partial class xp
    {
        public static NDarray frombuffer(byte[] buffer, Dtype dtype = null, int count = -1, int offset = 0)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.frombuffer(buffer, dtype?.CupyDtype, count, offset));
            else
                throw new NotSupportedException();
        }
    }
}
