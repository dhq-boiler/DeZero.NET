using Python.Runtime;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Text;

namespace DeZero.NET.Core
{
    public static class VRAMLeakDetector
    {
        private static readonly ConcurrentDictionary<string, List<ObjectMemoryInfo>> _allocTraces =
            new ConcurrentDictionary<string, List<ObjectMemoryInfo>>();

        private static readonly ConcurrentDictionary<IntPtr, string> _handleToStack =
            new ConcurrentDictionary<IntPtr, string>();

        private const double SIZE_CHANGE_THRESHOLD = 0.10;

        public static bool IsEnabled { get; set; } = false;

        public static int Iteration { get; set; } = 0;

        private class ObjectMemoryInfo
        {
            public WeakReference Reference { get; set; }
            public long InitialMemorySize { get; set; }  // 初期メモリサイズ
            public long CurrentMemorySize { get; set; }  // 現在のメモリサイズ
            public DateTime AllocationTime { get; set; }
            public List<(DateTime Time, long Size)> SizeHistory { get; set; } = new();
            public string ObjectType { get; set; }       // オブジェクトの型情報
            public int AllocatedIteration { get; set; }
        }

        public static void TrackAllocation(NDarray array, string location = null)
        {
            PythonObjectTracker.TrackPythonObject(array.PyObject, location);

            if (!IsEnabled) return;

            var stackTrace = new StackTrace(true);
            var trace = $"{location ?? "Unknown"}\n{stackTrace}";

            long memorySize = CalculateArrayMemorySize(array);
            var memInfo = new ObjectMemoryInfo
            {
                Reference = new WeakReference(array),
                InitialMemorySize = memorySize,
                CurrentMemorySize = memorySize,
                AllocationTime = DateTime.Now,
                ObjectType = GetDetailedArrayInfo(array),
                AllocatedIteration = Iteration
            };
            memInfo.SizeHistory.Add((DateTime.Now, memorySize));

            _allocTraces.AddOrUpdate(
                trace,
                new List<ObjectMemoryInfo> { memInfo },
                (_, list) => { list.Add(memInfo); return list; }
            );

            if (array.CupyNDarray != null)
            {
                _handleToStack[array.CupyNDarray.Handle] = trace;
            }

            using (Py.GIL())
            {
                using dynamic gc = Py.Import("gc");
                dynamic related = gc.get_referents(array.PyObject);
                foreach (dynamic obj in related)
                {
                    if (!obj.ToString().Contains("numpy") && obj.__class__.ToString().Equals("<class 'cupy.ndarray'>"))
                    {
                        PythonObjectTracker.TrackPythonObject(obj, location);
                    }
                }
            }
        }

        public static void TrackAllocation(Dtype dtype, string location = null)
        {
            if (!IsEnabled) return;

            if (dtype.PyObject is null) return;

            var stackTrace = new StackTrace(true);
            var trace = $"{location ?? "Unknown"}\n{stackTrace}";

            var memorySize = GetUnmanagedMemorySize(dtype.PyObject);
            var memInfo = new ObjectMemoryInfo
            {
                Reference = new WeakReference(dtype),
                InitialMemorySize = memorySize,
                CurrentMemorySize = memorySize,
                AllocationTime = DateTime.Now,
                ObjectType = GetDetailedArrayInfo(dtype),
                AllocatedIteration = Iteration
            };
            memInfo.SizeHistory.Add((DateTime.Now, memorySize));

            _allocTraces.AddOrUpdate(
                trace,
                new List<ObjectMemoryInfo> { memInfo },
                (_, list) => { list.Add(memInfo); return list; }
            );
        }

        public static void TrackAllocation(Flags flags, string location = null)
        {
            if (!IsEnabled) return;

            var stackTrace = new StackTrace(true);
            var trace = $"{location ?? "Unknown"}\n{stackTrace}";

            var memorySize = GetUnmanagedMemorySize(flags.PyObject);
            var memInfo = new ObjectMemoryInfo
            {
                Reference = new WeakReference(flags),
                InitialMemorySize = memorySize,
                CurrentMemorySize = memorySize,
                AllocationTime = DateTime.Now,
                ObjectType = GetDetailedArrayInfo(flags),
                AllocatedIteration = Iteration
            };
            memInfo.SizeHistory.Add((DateTime.Now, memorySize));

            _allocTraces.AddOrUpdate(
                trace,
                new List<ObjectMemoryInfo> { memInfo },
                (_, list) => { list.Add(memInfo); return list; }
            );
        }

        public static void TrackAllocation(Matrix matrix, string location = null)
        {
            if (!IsEnabled) return;

            var stackTrace = new StackTrace(true);
            var trace = $"{location ?? "Unknown"}\n{stackTrace}";

            var memorySize = GetUnmanagedMemorySize(matrix.PyObject);
            var memInfo = new ObjectMemoryInfo
            {
                Reference = new WeakReference(matrix),
                InitialMemorySize = memorySize,
                CurrentMemorySize = memorySize,
                AllocationTime = DateTime.Now,
                ObjectType = GetDetailedArrayInfo(matrix),
                AllocatedIteration = Iteration
            };
            memInfo.SizeHistory.Add((DateTime.Now, memorySize));

            _allocTraces.AddOrUpdate(
                trace,
                new List<ObjectMemoryInfo> { memInfo },
                (_, list) => { list.Add(memInfo); return list; }
            );
        }

        private static long GetUnmanagedMemorySize(PyObject pyobj)
        {
            if (pyobj == null) return 0;

            using (Py.GIL())
            {
                try
                {
                    using var sys = Py.Import("sys");
                    var directSize = sys.InvokeMethod("getsizeof", pyobj).As<long>();
                    return directSize;
                }
                catch (PythonException ex)
                {
                    Console.WriteLine($"Error getting memory size: {ex.Message}");
                    return 0;
                }
            }
        }

        public static void UpdateObjectSize(NDarray array)
        {
            if (!IsEnabled) return;

            foreach (var traceList in _allocTraces.Values)
            {
                foreach (var info in traceList)
                {
                    if (info.Reference.Target == array)
                    {
                        long newSize = CalculateArrayMemorySize(array);

                        // サイズ変更が閾値を超えた場合のみ記録
                        if (Math.Abs((double)(newSize - info.CurrentMemorySize) / info.CurrentMemorySize) > SIZE_CHANGE_THRESHOLD)
                        {
                            info.CurrentMemorySize = newSize;
                            info.SizeHistory.Add((DateTime.Now, newSize));
                        }
                        return;
                    }
                }
            }
        }

        private static long CalculateArrayMemorySize(NDarray array)
        {
            if (array.PyObject.__class__.ToString().Contains("NpzFile"))
            {
                return -1;
            }
            try
            {
                using (Py.GIL())
                {
                    // Get element size based on dtype
                    int elementSize = GetElementSize(array);

                    // Calculate total number of elements
                    long totalElements = 1;
                    using var shape = array.shape;
                    foreach (var dim in shape.Dimensions)
                    {
                        totalElements *= (long)dim;
                    }

                    // Calculate total memory size
                    return totalElements * elementSize;
                }
            }
            catch (Exception ex)
            {
                //Console.WriteLine($"Error calculating array size: {ex.Message}");
                return -1;
            }
        }

        private static int GetElementSize(NDarray array)
        {
            if (array.PyObject.__class__.ToString().Contains("NpzFile"))
            {
                return -1;
            }
            if (xp.isscalar(array))
            {
                return 4;  // default to 4 bytes for scalar types
            }
            using var array_dtype = array.dtype;
            string dtype = array_dtype.ToString().ToLower();
            return dtype switch
            {
                "float32" or "int32" => 4,
                "float64" or "int64" => 8,
                "float16" or "int16" => 2,
                "int8" or "uint8" => 1,
                "bool" => 1,
                _ => 4  // default to 4 bytes if unknown
            };
        }

        private static string GetDetailedArrayInfo(NDarray array)
        {
            if (array.PyObject.__class__.ToString().Contains("NpzFile"))
            {
                return "[NpzFile]";
            }

            try
            {
                using (Py.GIL())
                {
                    using var shape = array.shape;
                    using var dtype = array.dtype;
                    return $"[NDarray]Shape: {string.Join("x", shape)}, " +
                           $"dtype: {dtype}";
                }
            }
            catch (Exception ex)
            {
                return $"Error getting array info: {ex.Message}";
            }
        }

        private static string GetDetailedArrayInfo(Dtype dtype)
        {
            try
            {
                using (Py.GIL())
                {
                    return $"[Dtype]{dtype.ToString().ToLower()}";
                }
            }
            catch (Exception ex)
            {
                return $"Error getting array info: {ex.Message}";
            }
        }

        private static string GetDetailedArrayInfo(Flags flags)
        {
            try
            {
                using (Py.GIL())
                {
                    return $"[Flags]{flags.ToString().ToLower()}";
                }
            }
            catch (Exception ex)
            {
                return $"Error getting array info: {ex.Message}";
            }
        }

        private static string GetDetailedArrayInfo(Matrix matrix)
        {
            try
            {
                using (Py.GIL())
                {
                    return $"[Matrix]{matrix.ToString().ToLower()}";
                }
            }
            catch (Exception ex)
            {
                return $"Error getting array info: {ex.Message}";
            }
        }

        public static string GetAllocationReport(bool includeGrowthAnalysis = true)
        {
            if (!IsEnabled) return null;

            var sb = new StringBuilder();
            sb.AppendLine("=== VRAM Allocation Report ===\n");

            long totalLiveMemory = 0;
            int totalLiveObjects = 0;
            var locationStats = new List<(string Location, int LiveCount, long LiveMemory, List<ObjectGrowthInfo> GrowthInfo)>();

            foreach (var kvp in _allocTraces)
            {
                var liveObjects = kvp.Value.Where(info => info.Reference.IsAlive).ToList();
                if (liveObjects.Any())
                {
                    var liveCount = liveObjects.Count;
                    var liveMemory = liveObjects.Sum(info => info.CurrentMemorySize);

                    var growthInfo = liveObjects
                        .Where(info => info.SizeHistory.Count > 1)
                        .Select(info => new ObjectGrowthInfo
                        {
                            InitialSize = info.InitialMemorySize,
                            CurrentSize = info.CurrentMemorySize,
                            GrowthFactor = (double)info.CurrentMemorySize / info.InitialMemorySize,
                            Type = info.ObjectType,
                            GrowthHistory = info.SizeHistory
                        })
                        .ToList();

                    locationStats.Add((kvp.Key, liveCount, liveMemory, growthInfo));
                    totalLiveObjects += liveCount;
                    totalLiveMemory += liveMemory;
                }
            }

            // メモリ使用量の多い順にソート
            foreach (var stat in locationStats.OrderByDescending(s => s.LiveMemory))
            {
                sb.AppendLine($"Location: {stat.Location}");
                sb.AppendLine($"Live objects: {stat.LiveCount}");
                sb.AppendLine($"Current memory usage: {FormatMemorySize(stat.LiveMemory)}");

                if (includeGrowthAnalysis && stat.GrowthInfo.Any())
                {
                    sb.AppendLine("\nObject Growth Analysis:");
                    foreach (var growth in stat.GrowthInfo.OrderByDescending(g => g.GrowthFactor))
                    {
                        if (growth.GrowthFactor > 1.1) // 10%以上成長したオブジェクトのみ表示
                        {
                            sb.AppendLine($"  - {growth.Type}");
                            sb.AppendLine($"    Initial: {FormatMemorySize(growth.InitialSize)}");
                            sb.AppendLine($"    Current: {FormatMemorySize(growth.CurrentSize)}");
                            sb.AppendLine($"    Growth Factor: {growth.GrowthFactor:F2}x");

                            // 成長履歴のサマリー
                            if (growth.GrowthHistory.Count > 2)
                            {
                                var timeSpan = growth.GrowthHistory.Last().Time - growth.GrowthHistory.First().Time;
                                sb.AppendLine($"    Growth Period: {timeSpan.TotalMinutes:F1} minutes");
                                sb.AppendLine($"    Growth Steps: {growth.GrowthHistory.Count}");
                            }
                        }
                    }
                }
                sb.AppendLine();
            }

            sb.AppendLine("=== Objects instantiated in the latest iteration ===");

            long allocatedMemoryOnLatestIteration = 0;
            var _locationStats = new List<(string Location, int LiveCount, long LiveMemory, int AllocatedIteration, List<ObjectGrowthInfo> GrowthInfo)>();

            foreach (var kvp in _allocTraces)
            {
                var liveObjects = kvp.Value.Where(info => info.Reference.IsAlive && info.AllocatedIteration == kvp.Value.Max(x => x.AllocatedIteration)).ToList();
                if (liveObjects.Any())
                {
                    var liveCount = liveObjects.Count;
                    var liveMemory = liveObjects.Sum(info => info.CurrentMemorySize);
                    var allocatedIteration = liveObjects.First().AllocatedIteration;

                    var growthInfo = liveObjects
                        .Where(info => info.SizeHistory.Count > 1)
                        .Select(info => new ObjectGrowthInfo
                        {
                            InitialSize = info.InitialMemorySize,
                            CurrentSize = info.CurrentMemorySize,
                            GrowthFactor = (double)info.CurrentMemorySize / info.InitialMemorySize,
                            Type = info.ObjectType,
                            GrowthHistory = info.SizeHistory
                        })
                        .ToList();

                    allocatedMemoryOnLatestIteration += liveMemory;
                    _locationStats.Add((kvp.Key, liveCount, liveMemory, allocatedIteration, growthInfo));
                }
            }

            foreach (var stat in _locationStats.OrderByDescending(s => s.LiveMemory))
            {
                sb.AppendLine($"ALLOCATED LATEST ITERATION");
                sb.AppendLine($"Location: {stat.Location}");
                sb.AppendLine($"Live objects: {stat.LiveCount}");
                sb.AppendLine($"Current memory usage: {FormatMemorySize(stat.LiveMemory)}");

                if (includeGrowthAnalysis && stat.GrowthInfo.Any())
                {
                    sb.AppendLine("\nObject Growth Analysis:");
                    foreach (var growth in stat.GrowthInfo.OrderByDescending(g => g.GrowthFactor))
                    {
                        if (growth.GrowthFactor > 1.1) // 10%以上成長したオブジェクトのみ表示
                        {
                            sb.AppendLine($"  - {growth.Type}");
                            sb.AppendLine($"    Initial: {FormatMemorySize(growth.InitialSize)}");
                            sb.AppendLine($"    Current: {FormatMemorySize(growth.CurrentSize)}");
                            sb.AppendLine($"    Growth Factor: {growth.GrowthFactor:F2}x");

                            // 成長履歴のサマリー
                            if (growth.GrowthHistory.Count > 2)
                            {
                                var timeSpan = growth.GrowthHistory.Last().Time - growth.GrowthHistory.First().Time;
                                sb.AppendLine($"    Growth Period: {timeSpan.TotalMinutes:F1} minutes");
                                sb.AppendLine($"    Growth Steps: {growth.GrowthHistory.Count}");
                            }
                        }
                    }
                }
                sb.AppendLine();
            }

            // サマリー情報
            sb.AppendLine("=== Summary ===");
            sb.AppendLine($"Total live objects: {totalLiveObjects}");
            sb.AppendLine($"Total memory used by tracked objects: {FormatMemorySize(totalLiveMemory)}");
            sb.AppendLine(
                $"Total memory allocated on latest iteration: {FormatMemorySize(allocatedMemoryOnLatestIteration)}");

            // メモリプール情報
            using (Py.GIL())
            {
                try
                {
                    using dynamic cupy = Py.Import("cupy");
                    var mempool = cupy.get_default_memory_pool();
                    var usedBytes = (long)mempool.used_bytes();
                    var totalBytes = (long)mempool.total_bytes();

                    sb.AppendLine($"\nMemory Pool Status:");
                    sb.AppendLine($"Used: {FormatMemorySize(usedBytes)}");
                    sb.AppendLine($"Total: {FormatMemorySize(totalBytes)}");
                    sb.AppendLine($"Fragmentation: {((double)totalBytes / usedBytes - 1) * 100:F1}%");
                }
                catch (Exception ex)
                {
                    sb.AppendLine($"\nError getting memory pool info: {ex.Message}");
                }
            }

            return sb.ToString();
        }

        private class ObjectGrowthInfo
        {
            public long InitialSize { get; set; }
            public long CurrentSize { get; set; }
            public double GrowthFactor { get; set; }
            public string Type { get; set; }
            public List<(DateTime Time, long Size)> GrowthHistory { get; set; }
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

        // 定期的なメモリ使用量のロギング用
        public static void StartMemoryMonitoring(int intervalMs = 1000)
        {
            if (!IsEnabled) return;

            var timer = new Timer(_ =>
            {
                var usage = GpuMemoryMonitor.Instance.GetCurrentMemoryUsage();
                var handleCount = _handleToStack.Count;
                var liveObjectsCount = _allocTraces.Values.Sum(list => list.Count(info => info.Reference.IsAlive));
                var liveObjectsMemory = _allocTraces.Values.Sum(list =>
                    list.Where(info => info.Reference.IsAlive)
                        .Sum(info => info.CurrentMemorySize));

                Console.WriteLine($"[{DateTime.Now:HH:mm:ss.fff}] " +
                    $"VRAM: {usage}MB, " +
                    $"Active handles: {handleCount}, " +
                    $"Live objects: {liveObjectsCount} " +
                    $"({FormatMemorySize(liveObjectsMemory)})");
            }, null, 0, intervalMs);
        }

        private static long previousPotentialLeak = 0;

        // メモリリークの可能性がある箇所を検出
        public static void DetectPotentialLeaks()
        {
            if (!IsEnabled) return;

            using (Py.GIL())
            {
                try
                {
                    using dynamic cupy = Py.Import("cupy");
                    var mempool = cupy.get_default_memory_pool();
                    var usedBytes = (long)mempool.used_bytes();
                    var totalBytes = (long)mempool.total_bytes();

                    Console.WriteLine($"Memory pool status:");
                    Console.WriteLine($"Used: {usedBytes / 1024 / 1024}MB");
                    Console.WriteLine($"Total: {totalBytes / 1024 / 1024}MB");

                    // デバイス情報の取得
                    try
                    {
                        dynamic device = cupy.cuda.Device();
                        dynamic attributes = device.attributes;
                        var totalDeviceMemory = (long)device.mem_info[1];  // 合計メモリ
                        var freeDeviceMemory = (long)device.mem_info[0];   // 空きメモリ

                        Console.WriteLine("\nDevice Memory Info:");
                        Console.WriteLine($"Total Memory: {totalDeviceMemory / 1024 / 1024}MB");
                        Console.WriteLine($"Free Memory: {freeDeviceMemory / 1024 / 1024}MB");
                        Console.WriteLine($"Used Memory: {(totalDeviceMemory - freeDeviceMemory) / 1024 / 1024}MB");
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Failed to get device info: {ex.Message}");
                    }

                    // メモリリークの検出
                    Console.WriteLine("\nChecking for memory leaks...");
                    var beforeCleanup = usedBytes;
                    mempool.free_all_blocks();
                    GC.Collect(2, GCCollectionMode.Forced, true);
                    GC.WaitForPendingFinalizers();

                    var afterCleanup = (long)mempool.used_bytes();
                    var potentialLeak = afterCleanup;

                    var diff = potentialLeak - previousPotentialLeak;
                    if (potentialLeak > 0)
                    {
                        Console.WriteLine($"Potential memory leak detected: {FormatMemorySize(potentialLeak)} ({(diff < 0 ? "▲" : "+")}{FormatMemorySize(potentialLeak - previousPotentialLeak)})");
                    }

                    previousPotentialLeak = potentialLeak;

                    // メモリの断片化状態を確認
                    Console.WriteLine("\nMemory Fragmentation Status:");
                    var blocksFreed = (beforeCleanup - afterCleanup) / 1024 / 1024;
                    Console.WriteLine($"Memory freed by cleanup: {blocksFreed}MB");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error during leak detection: {ex.Message}");
                }
            }
        }
    }
}
