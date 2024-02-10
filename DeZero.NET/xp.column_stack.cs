using Cupy;
using Numpy;

namespace DeZero.NET
{
    public static partial class xp
    {
        public static NDarray column_stack(params NDarray[] tup)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.column_stack(tup));
            }
            else
            {
                return new NDarray(np.column_stack(tup));
            }
		}
    }
}
