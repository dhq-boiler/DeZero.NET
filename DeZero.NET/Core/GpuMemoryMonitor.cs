using DeZero.NET.Log;
using Python.Runtime;
using System.Diagnostics;

namespace DeZero.NET.Core
{
    public class GpuMemoryMonitor : IDisposable
    {
        private static volatile GpuMemoryMonitor _instance;
        private static readonly object _lock = new object();
        public static bool IsEnabled { get; set; } = false;
        public static LogLevel LogLevel { get; set; } = LogLevel.Info;
        public static bool IsVerbose { get; set; } = false;

        private const double WARNING_THRESHOLD = 0.50;

        public static GpuMemoryMonitor Instance
        {
            get
            {
                if (_instance == null)
                {
                    lock (_lock)
                    {
                        if (_instance == null)
                        {
                            _instance = new GpuMemoryMonitor();
                        }
                    }
                }
                return _instance;
            }
        }

        private readonly ILogger _logger;
        private readonly object _pythonLock = new object();
        private PyObject _cupyModule;
        private PyObject _mempoolObject;
        private volatile bool _isDisposed;

        private GpuMemoryMonitor()
        {
            _logger = new ConsoleLogger(LogLevel, isVerbose: IsVerbose);
            InitializePythonObjects();
        }

        private void InitializePythonObjects()
        {
            lock (_pythonLock)
            {
                try
                {
                    using (Py.GIL())
                    {
                        DisposePythonObjects();

                        _cupyModule = Py.Import("cupy");
                        if (_cupyModule == null)
                        {
                            throw new InvalidOperationException("Failed to import cupy");
                        }

                        _mempoolObject = _cupyModule.InvokeMethod("get_default_memory_pool");
                        if (_mempoolObject == null)
                        {
                            throw new InvalidOperationException("Failed to get memory pool");
                        }
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError($"Failed to initialize Python objects: {ex.Message}");
                    DisposePythonObjects();
                }
            }
        }

        public void LogMemoryUsage(string location, bool verbose = false)
        {
            if (!IsEnabled) return;

            if (_isDisposed) return;

            verbose |= IsVerbose;

            try
            {
                using (Py.GIL())
                {
                    var (totalMemory, usedMemory) = CleanupMemory(location, verbose);
                    var dicCount = LogCupyObjects(ndarray_only: true);
                    LogMemoryStats(location, totalMemory, usedMemory);

                    //コンソールをクリア
                    for (int i = 0; i < Console.WindowHeight - 5 - dicCount; i++)
                    {
                        Console.WriteLine();
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError($"Failed to log memory usage: {ex.Message}");
            }
        }

        public (long totalMemory, long usedMemory) CleanupMemory(string location = null, bool verbose = false)
        {
            var (totalMemory, usedMemory, freeMemory) = GetMemoryInfo();
            //if ((double)usedMemory / totalMemory > WARNING_THRESHOLD)
            //{
                HandleHighMemoryUsage(location, usedMemory, totalMemory, verbose);
            //}

            return (totalMemory, usedMemory);
        }

        private int LogCupyObjects(bool ndarray_only = false)
        {
            using dynamic gc = Py.Import("gc");
            using dynamic sys = Py.Import("sys");

            using dynamic cupyObjects = gc.get_objects();

            var dic = new Dictionary<string, Dictionary<string, long>>();
            var ndarray_dic = new Dictionary<string, Dictionary<string, long>>();

            foreach (var obj in cupyObjects)
            {
                switch (obj.__class__.ToString())
                {
                    case "<class 'cupy.ndarray'>":
                    {
                        var type_name = "cupy.ndarray";
                        var size = obj.nbytes;
                        if (dic.ContainsKey(type_name))
                        {
                            dic[type_name]["count"] += 1;
                            dic[type_name]["total_size"] += size.As<long>();
                        }
                        else
                        {
                            dic[type_name] = new Dictionary<string, long>();
                            dic[type_name]["count"] = 1;
                            dic[type_name]["total_size"] = size.As<long>();
                        }

                        var shapeStr = obj.shape.ToString();
                        if (ndarray_dic.ContainsKey(shapeStr))
                        {
                            ndarray_dic[shapeStr]["count"] += 1;
                            ndarray_dic[shapeStr]["total_size"] += size.As<long>();
                        }
                        else
                        {
                            ndarray_dic[shapeStr] = new Dictionary<string, long>();
                            ndarray_dic[shapeStr]["count"] = 1;
                            ndarray_dic[shapeStr]["total_size"] = size.As<long>();
                        }
                    }
                        break;
                    case "<class 'cupy.cuda.memory._Chunk'>":
                    {
                        var type_name = "cupy.cuda.memory._Chunk";
                        var directSize = sys.InvokeMethod("getsizeof", obj).As<long>();
                        if (dic.ContainsKey(type_name))
                        {
                            dic[type_name]["count"] += 1;
                            dic[type_name]["total_size"] += directSize;
                        }
                        else
                        {
                            dic[type_name] = new Dictionary<string, long>();
                            dic[type_name]["count"] = 1;
                            dic[type_name]["total_size"] = directSize;
                        }
                    }
                        break;
                    case "<class 'cupy.cuda.memory.MemoryPointer'>":
                    {
                        var type_name = "cupy.cuda.memory.MemoryPointer";
                        var directSize = sys.InvokeMethod("getsizeof", obj).As<long>();
                        if (dic.ContainsKey(type_name))
                        {
                            dic[type_name]["count"] += 1;
                            dic[type_name]["total_size"] += directSize;
                        }
                        else
                        {
                            dic[type_name] = new Dictionary<string, long>();
                            dic[type_name]["count"] = 1;
                            dic[type_name]["total_size"] = directSize;
                        }
                    }
                        break;
                    case "<class 'tuple'>":
                    {
                        var type_name = "tuple";
                        var directSize = sys.InvokeMethod("getsizeof", obj).As<long>();
                        if (dic.ContainsKey(type_name))
                        {
                            dic[type_name]["count"] += 1;
                            dic[type_name]["total_size"] += directSize;
                        }
                        else
                        {
                            dic[type_name] = new Dictionary<string, long>();
                            dic[type_name]["count"] = 1;
                            dic[type_name]["total_size"] = directSize;
                        }
                    }
                        break;
                    case "<class 'cupy._core._kernel._ArgInfo'>":
                    {
                        var type_name = "cupy._core._kernel._ArgInfo";
                        var directSize = sys.InvokeMethod("getsizeof", obj).As<long>();
                        if (dic.ContainsKey(type_name))
                        {
                            dic[type_name]["count"] += 1;
                            dic[type_name]["total_size"] += directSize;
                        }
                        else
                        {
                            dic[type_name] = new Dictionary<string, long>();
                            dic[type_name]["count"] = 1;
                            dic[type_name]["total_size"] = directSize;
                        }
                    }
                        break;
                    case "<class 'cupy.cuda.function.Module'>":
                    {
                        var type_name = "cupy.cuda.function.Module";
                        var directSize = sys.InvokeMethod("getsizeof", obj).As<long>();
                        if (dic.ContainsKey(type_name))
                        {
                            dic[type_name]["count"] += 1;
                            dic[type_name]["total_size"] += directSize;
                        }
                        else
                        {
                            dic[type_name] = new Dictionary<string, long>();
                            dic[type_name]["count"] = 1;
                            dic[type_name]["total_size"] = directSize;
                        }
                    }
                        break;
                    case "<class 'cupy.cuda.function.Function'>":
                    {
                        var type_name = "cupy.cuda.function.Function";
                        var directSize = sys.InvokeMethod("getsizeof", obj).As<long>();
                        if (dic.ContainsKey(type_name))
                        {
                            dic[type_name]["count"] += 1;
                            dic[type_name]["total_size"] += directSize;
                        }
                        else
                        {
                            dic[type_name] = new Dictionary<string, long>();
                            dic[type_name]["count"] = 1;
                            dic[type_name]["total_size"] = directSize;
                        }
                    }
                        break;
                    case "<class 'builtin_function_or_method'>":
                    {
                        var type_name = "builtin_function_or_method";
                        var directSize = sys.InvokeMethod("getsizeof", obj).As<long>();
                        if (dic.ContainsKey(type_name))
                        {
                            dic[type_name]["count"] += 1;
                            dic[type_name]["total_size"] += directSize;
                        }
                        else
                        {
                            dic[type_name] = new Dictionary<string, long>();
                            dic[type_name]["count"] = 1;
                            dic[type_name]["total_size"] = directSize;
                        }
                    }
                        break;
                    case "<class 'cupy.cuda.memory.__pyx_scope_struct____init__'>":
                    {
                        var type_name = "cupy.cuda.memory.__pyx_scope_struct____init__";
                        var directSize = sys.InvokeMethod("getsizeof", obj).As<long>();
                        if (dic.ContainsKey(type_name))
                        {
                            dic[type_name]["count"] += 1;
                            dic[type_name]["total_size"] += directSize;
                        }
                        else
                        {
                            dic[type_name] = new Dictionary<string, long>();
                            dic[type_name]["count"] = 1;
                            dic[type_name]["total_size"] = directSize;
                        }
                    }
                        break;
                    case "<class 'list'>":
                    {
                        var type_name = "list";
                        var directSize = sys.InvokeMethod("getsizeof", obj).As<long>();
                        if (dic.ContainsKey(type_name))
                        {
                            dic[type_name]["count"] += 1;
                            dic[type_name]["total_size"] += directSize;
                        }
                        else
                        {
                            dic[type_name] = new Dictionary<string, long>();
                            dic[type_name]["count"] = 1;
                            dic[type_name]["total_size"] = directSize;
                        }
                    }
                        break;
                    case "<class 'getset_descriptor'>":
                    {
                        var type_name = "getset_descriptor";
                        var directSize = sys.InvokeMethod("getsizeof", obj).As<long>();
                        if (dic.ContainsKey(type_name))
                        {
                            dic[type_name]["count"] += 1;
                            dic[type_name]["total_size"] += directSize;
                        }
                        else
                        {
                            dic[type_name] = new Dictionary<string, long>();
                            dic[type_name]["count"] = 1;
                            dic[type_name]["total_size"] = directSize;
                        }
                    }
                        break;
                    case "<class 'weakref.ReferenceType'>":
                    {
                        var type_name = "weakref.ReferenceType";
                        var directSize = sys.InvokeMethod("getsizeof", obj).As<long>();
                        if (dic.ContainsKey(type_name))
                        {
                            dic[type_name]["count"] += 1;
                            dic[type_name]["total_size"] += directSize;
                        }
                        else
                        {
                            dic[type_name] = new Dictionary<string, long>();
                            dic[type_name]["count"] = 1;
                            dic[type_name]["total_size"] = directSize;
                        }
                    }
                        break;
                    case "<class 'StgDict'>":
                    {
                        var type_name = "StgDict";
                        var directSize = sys.InvokeMethod("getsizeof", obj).As<long>();
                        if (dic.ContainsKey(type_name))
                        {
                            dic[type_name]["count"] += 1;
                            dic[type_name]["total_size"] += directSize;
                        }
                        else
                        {
                            dic[type_name] = new Dictionary<string, long>();
                            dic[type_name]["count"] = 1;
                            dic[type_name]["total_size"] = directSize;
                        }
                    }
                        break;
                    case "<class 'dict'>":
                    {
                        var type_name = "dict";
                        var directSize = sys.InvokeMethod("getsizeof", obj).As<long>();
                        if (dic.ContainsKey(type_name))
                        {
                            dic[type_name]["count"] += 1;
                            dic[type_name]["total_size"] += directSize;
                        }
                        else
                        {
                            dic[type_name] = new Dictionary<string, long>();
                            dic[type_name]["count"] = 1;
                            dic[type_name]["total_size"] = directSize;
                        }
                    }
                        break;
                    case "<class 'cython_function_or_method'>":
                    {
                        var type_name = "cython_function_or_method";
                        var directSize = sys.InvokeMethod("getsizeof", obj).As<long>();
                        if (dic.ContainsKey(type_name))
                        {
                            dic[type_name]["count"] += 1;
                            dic[type_name]["total_size"] += directSize;
                        }
                        else
                        {
                            dic[type_name] = new Dictionary<string, long>();
                            dic[type_name]["count"] = 1;
                            dic[type_name]["total_size"] = directSize;
                        }
                    }
                        break;
                    case "<class 'cell'>":
                    {
                        var type_name = "cell";
                        var directSize = sys.InvokeMethod("getsizeof", obj).As<long>();
                        if (dic.ContainsKey(type_name))
                        {
                            dic[type_name]["count"] += 1;
                            dic[type_name]["total_size"] += directSize;
                        }
                        else
                        {
                            dic[type_name] = new Dictionary<string, long>();
                            dic[type_name]["count"] = 1;
                            dic[type_name]["total_size"] = directSize;
                        }
                    }
                        break;
                    default:
                        break;
                }
            }

            if (ndarray_only)
            {
                foreach (var (shape, values) in ndarray_dic.OrderByDescending(kvp => kvp.Value["total_size"]))
                {
                    _logger.LogDebug(
                        $"{shape,-50}:\t{values["count"]} objects,\t{FormatMemorySize(values["total_size"])}");
                }

                return ndarray_dic.Count;
            }
            else
            {
                foreach (var (type_name, values) in dic.OrderByDescending(kvp => kvp.Value["total_size"]))
                {
                    _logger.LogDebug(
                        $"{type_name,-50}:\t{values["count"]} objects,\t{FormatMemorySize(values["total_size"])}");
                }
                return dic.Count;
            }
        }

        private static string FormatMemorySize(long bytes)
        {
            string[] suffixes = { "B", "KB", "MB", "GB", "TB" };
            int counter = 0;
            decimal number = bytes;
            while (Math.Round(number / 1024) >= 1)
            {
                number /= 1024;
                counter++;
            }
            return $"{number:n2} {suffixes[counter]}";
        }

        private (long Total, long Used, long Free) GetMemoryInfo()
        {
            if (_mempoolObject is null)
            {
                InitializePythonObjects();
            }

            lock (_pythonLock)
            {
                try
                {
                    using var totalBytes = _mempoolObject.InvokeMethod("total_bytes");
                    using var usedBytes = _mempoolObject.InvokeMethod("used_bytes");

                    var totalMemory = Parse(totalBytes) / (1024 * 1024);
                    var usedMemory = Parse(usedBytes) / (1024 * 1024);
                    return (totalMemory, usedMemory, totalMemory - usedMemory);
                }
                catch (Exception ex)
                {
                    _logger.LogError($"Failed to get memory info: {ex.Message}");
                    InitializePythonObjects();
                    return (0, 0, 0);
                }
            }
        }

        private long Parse(PyObject pyObject)
        {
            try
            {
                if (pyObject == null) return 0;
                using var pyLong = new PyInt(pyObject);
                return pyLong.ToInt64();
            }
            catch
            {
                return 0;
            }
        }

        public long GetCurrentMemoryUsage()
        {
            lock (_pythonLock)
            {
                try
                {
                    using (Py.GIL())
                    {
                        if (_mempoolObject != null)
                        {
                            using var usedBytes = _mempoolObject.InvokeMethod("used_bytes");
                            return Parse(usedBytes) / (1024 * 1024); // バイトからMBに変換
                        }
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError($"Failed to get current memory usage: {ex.Message}");
                    InitializePythonObjects();
                }
                return 0;
            }
        }

        private void LogMemoryStats(string location, long totalMemory, long usedMemory)
        {
            if (!IsEnabled) return;

            _logger.LogDebug($"""
            Location: {location}
            Total Memory: {totalMemory:N0} MB
            Used Memory: {usedMemory:N0} MB
            Free Memory: {(totalMemory - usedMemory):N0} MB
            """);
        }

        private void HandleHighMemoryUsage(string location, long usedMemory, long totalMemory, bool verbose)
        {
            if (verbose)
            {
                _logger.LogWarning($"WARNING: High GPU memory usage detected at {location}! ({usedMemory:N0}/{totalMemory:N0} MB)");
            }

            GC.Collect(2, GCCollectionMode.Forced, true);
            Finalizer.Instance.Collect();
            ForceMemoryPool();
        }

        public static void ForceMemoryPool()
        {
            try
            {
                using (Py.GIL())
                {
                    if (Instance._mempoolObject != null)
                    {
                        using var result = Instance._mempoolObject.InvokeMethod("free_all_blocks");
                    }
                }
            }
            catch (Exception ex)
            {
                Instance._logger.LogError($"Failed to force memory pool: {ex.Message}");
            }
        }

        private void DisposePythonObjects()
        {
            _mempoolObject?.Dispose();
            _mempoolObject = null;
            _cupyModule?.Dispose();
            _cupyModule = null;
        }

        public void Dispose()
        {
            if (_isDisposed) return;

            _isDisposed = true;

            lock (_pythonLock)
            {
                try
                {
                    using (Py.GIL())
                    {
                        DisposePythonObjects();
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError($"Error during disposal: {ex.Message}");
                }
            }

            _instance = null;
        }
    }

    // 使用例を示すための拡張メソッド
    public static class MemoryMonitoringExtensions
    {
        public static IDisposable TrackMemory(this object _, string location, bool verbose = false)
        {
            return new MemoryTrackingScope(location, verbose);
        }
    }

    public class MemoryTrackingScope : IDisposable
    {
        private readonly string _location;
        private readonly bool _verbose;
        private readonly Stopwatch _stopwatch;

        public MemoryTrackingScope(string location, bool verbose = false)
        {
            _location = location;
            _verbose = verbose;
            _stopwatch = Stopwatch.StartNew();
            GpuMemoryMonitor.Instance.LogMemoryUsage($"{_location} - Start", _verbose);
        }

        public void Dispose()
        {
            _stopwatch.Stop();
            GpuMemoryMonitor.Instance.LogMemoryUsage($"{_location} - End ({_stopwatch.ElapsedMilliseconds}ms)", _verbose);
        }
    }
}
