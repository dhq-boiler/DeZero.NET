using Cupy;
using Numpy;
using System.Reflection;

namespace DeZero.NET
{
    public static partial class xp
    {
        public static void Initialize()
        {
            MethodInfo method;
            if (Gpu.Available && Gpu.Use)
            {
                method = typeof(cp).GetMethod("ReInitializeLazySelf", BindingFlags.NonPublic | BindingFlags.Static);
                method.Invoke(null, []);
            }
            method = typeof(np).GetMethod("ReInitializeLazySelf", BindingFlags.NonPublic | BindingFlags.Static);
            method.Invoke(null, []);
        }
    }
}
