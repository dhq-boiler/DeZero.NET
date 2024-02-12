using Cupy;
using Numpy;
using System.Reflection;

namespace DeZero.NET
{
    public static partial class xp
    {
        public static void Initialize()
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                var method = typeof(cp).GetMethod("ReInitializeLazySelf", BindingFlags.NonPublic | BindingFlags.Static);
                method.Invoke(null, []);
            }
            else
            {
                var method = typeof(np).GetMethod("ReInitializeLazySelf", BindingFlags.NonPublic | BindingFlags.Static);
                method.Invoke(null, []);
            }
        }
    }
}
