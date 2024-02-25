using Cupy;
using Numpy;
using Python.Runtime;

namespace DeZero.NET
{
    public static partial class ctypes
    {
        private static readonly Lazy<PyObject> _lazy_self = new Lazy<PyObject>(() =>
        {
            if (Gpu.Available && Gpu.Use)
            {
                var x = cp.self; // <-- make sure np initializes the python engine
                var mod = Py.Import("ctypes");
                return mod;
            }
            else
            {
                var x = np.self; // <-- make sure np initializes the python engine
                var mod = Py.Import("ctypes");
                return mod;
            }
        });

        public static PyObject self => _lazy_self.Value;

        public static dynamic dynamic_self => self;
        private static bool IsInitialized => self != null;

        public static PyObject data => self.GetAttr("data");

        public static void Dispose()
        {
            self?.Dispose();
        }
    }
}
