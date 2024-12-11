using Python.Runtime;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Text;

namespace DeZero.NET.Core
{

    public static class PythonObjectTracker
    {
        private class TrackedPythonObject
        {
            public WeakReference<PyObject> Reference { get; set; }
            public long Size { get; set; }
            public string Type { get; set; }
            public DateTime CreationTime { get; set; }
            public string StackTrace { get; set; }
            public Dictionary<string, object> Attributes { get; set; }
            public List<(DateTime Time, long Size)> SizeHistory { get; set; }
            public bool IsArrayBuffer { get; set; }
        }

        private static readonly ConcurrentDictionary<IntPtr, TrackedPythonObject> _trackedObjects =
            new ConcurrentDictionary<IntPtr, TrackedPythonObject>();

        private static readonly HashSet<string> _cupyTypes = new HashSet<string>
        {
            "<class 'cupy.ndarray'>",
            "<class 'cupy.cuda.memory._Chunk'>",
            "<class 'cupy.cuda.memory.MemoryPointer'>",
            "<class 'cupy._core._kernel._ArgInfo'>",
            "<class 'cupy.cuda.function.Module'>",
            "<class 'cupy.cuda.function.Function'>"
        };

        public static bool IsEnabled { get; set; } = false;

        public static string DebugDetectingShape { get; set; } = string.Empty;

        public static void TrackPythonObject(dynamic obj, string location = null)
        {
            if (!IsEnabled) return;

            if (obj == null) return;

            using (Py.GIL())
            {
                try
                {
                    // オブジェクトの型を確認
                    var objType = obj.__class__.ToString();
                    if (!_cupyTypes.Contains(objType))
                    {
                        return; // 追跡対象外の型
                    }

                    if (!string.IsNullOrEmpty(DebugDetectingShape) && objType.Contains("cupy.ndarray")
                                                                   && obj.shape.ToString().Equals(DebugDetectingShape))
                    {
                        Debugger.Break();
                    }

                    var handle = obj.Handle;
                    var stackTrace = new StackTrace(true);
                    var traceLoc = location ?? "Unknown";

                    var trackedObj = new TrackedPythonObject
                    {
                        Reference = new WeakReference<PyObject>(obj),
                        Type = objType,
                        CreationTime = DateTime.Now,
                        StackTrace = $"{traceLoc}\n{stackTrace}",
                        Attributes = new Dictionary<string, object>(),
                        SizeHistory = new List<(DateTime, long)>(),
                        IsArrayBuffer = IsCupyArrayBuffer(obj)
                    };

                    // サイズ計算
                    trackedObj.Size = CalculatePythonObjectSize(obj);
                    trackedObj.SizeHistory.Add((DateTime.Now, trackedObj.Size));

                    // 追加属性の収集
                    CollectObjectAttributes(obj, trackedObj);

                    _trackedObjects.AddOrUpdate(handle, trackedObj, new Func<IntPtr, TrackedPythonObject, TrackedPythonObject>((_, existing) =>
                    {
                        // 既存オブジェクトの更新
                        existing.Size = trackedObj.Size;
                        existing.SizeHistory.Add((DateTime.Now, trackedObj.Size));
                        return existing;
                    }));

                    // メモリ使用量が閾値を超えた場合の警告
                    if (trackedObj.Size > 1024 * 1024 * 100) // 100MB
                    {
                        LogLargeAllocation(trackedObj);
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error tracking Python object: {ex.Message}");
                }
            }
        }

        public static void UnTrackPythonObject(dynamic obj, string location = null)
        {
            if (!IsEnabled) return;

            if (obj.Handle == IntPtr.Zero) return;

            using (Py.GIL())
            {
                try
                {
                    // オブジェクトの型を確認
                    var objType = obj.__class__.ToString();
                    if (!_cupyTypes.Contains(objType))
                    {
                        return; // 追跡対象外の型
                    }

                    var handle = obj.Handle;

                    // オブジェクトが追跡されているか確認
                    if (_trackedObjects.TryRemove(handle, out TrackedPythonObject trackedObj))
                    {
                        // 最終状態のログを記録
                        var finalSize = CalculatePythonObjectSize(obj);
                        trackedObj.SizeHistory.Add((DateTime.Now, finalSize));

                        var sb = new StringBuilder();
                        sb.AppendLine($"Python object untracked:");
                        sb.AppendLine($"Type: {trackedObj.Type}");
                        sb.AppendLine($"Final Size: {FormatSize(finalSize)}");
                        sb.AppendLine($"Lifetime: {(DateTime.Now - trackedObj.CreationTime).TotalSeconds:F2} seconds");
                        sb.AppendLine($"Untrack Location: {location ?? "Unknown"}");

                        // メモリ使用量の履歴を分析
                        if (trackedObj.SizeHistory.Count > 1)
                        {
                            var initialSize = trackedObj.SizeHistory.First().Size;
                            var peakSize = trackedObj.SizeHistory.Max(x => x.Size);
                            sb.AppendLine($"Initial Size: {FormatSize(initialSize)}");
                            sb.AppendLine($"Peak Size: {FormatSize(peakSize)}");
                        }

                        if (finalSize > 1024 * 1024 * 100) // 100MB
                        {
                            sb.AppendLine($"Warning: Large object being untracked");
                            sb.AppendLine($"Original Stack Trace:\n{trackedObj.StackTrace}");
                        }

                        Console.WriteLine(sb.ToString());

                        // WeakReferenceをクリア
                        trackedObj.Reference.SetTarget(null);
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error untracking Python object: {ex.Message}");
                }
            }
        }
        private static bool IsCupyArrayBuffer(dynamic obj)
        {
            try
            {
                return obj.__class__.ToString() == "<class 'cupy.ndarray'>" &&
                       obj.HasAttr("__cuda_array_interface__");
            }
            catch
            {
                return false;
            }
        }

        private static long CalculatePythonObjectSize(dynamic obj)
        {
            using (Py.GIL())
            {
                try
                {
                    using var sys = Py.Import("sys");
                    long baseSize = sys.InvokeMethod("getsizeof", obj).As<long>();

                    // cupyアレイの場合、実際のメモリサイズを計算
                    if (obj.__class__.ToString() == "<class 'cupy.ndarray'>")
                    {
                        using var array = obj.As<dynamic>();
                        var shape = array.shape;
                        var dtype = array.dtype;
                        long totalElements = 1;
                        foreach (var dim in shape)
                        {
                            totalElements *= (long)dim;
                        }

                        // dtypeのサイズを取得
                        var itemsize = (long)dtype.itemsize;
                        return totalElements * itemsize;
                    }

                    return baseSize;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error calculating object size: {ex.Message}");
                    return 0;
                }
            }
        }

        private static void CollectObjectAttributes(dynamic obj, TrackedPythonObject trackedObj)
        {
            if (obj.__class__.ToString() == "<class 'cupy.ndarray'>")
            {
                using dynamic array = obj.As<dynamic>();
                trackedObj.Attributes["shape"] = array.shape.ToString();
                trackedObj.Attributes["dtype"] = array.dtype.ToString();
                if (obj.HasAttr("__cuda_array_interface__"))
                {
                    var  _interface = array.__cuda_array_interface__;
                    trackedObj.Attributes["data_ptr"] =  _interface["data"][0];
                }
            }
        }

        private static void LogLargeAllocation(TrackedPythonObject obj)
        {
            var sb = new StringBuilder();
            sb.AppendLine($"Large Python object allocation detected!");
            sb.AppendLine($"Type: {obj.Type}");
            sb.AppendLine($"Size: {FormatSize(obj.Size)}");
            sb.AppendLine($"Creation Time: {obj.CreationTime}");
            sb.AppendLine($"Stack Trace:\n{obj.StackTrace}");

            if (obj.Attributes.Count > 0)
            {
                sb.AppendLine("Attributes:");
                foreach (var attr in obj.Attributes)
                {
                    sb.AppendLine($"  {attr.Key}: {attr.Value}");
                }
            }

            Console.WriteLine(sb.ToString());
        }

        public static string GenerateReport()
        {
            if (!IsEnabled) return string.Empty;

            var sb = new StringBuilder();
            sb.AppendLine("=== Python Objects Tracking Report ===\n");

            var grouped = _trackedObjects
                .Where(kvp => IsObjectAlive(kvp.Value))
                .GroupBy(kvp => kvp.Value.Type);

            foreach (var group in grouped.OrderByDescending(g => g.Sum(kvp => kvp.Value.Size)))
            {
                var count = group.Count();
                var totalSize = group.Sum(kvp => kvp.Value.Size);
                var avgSize = count > 0 ? totalSize / count : 0;

                sb.AppendLine($"Type: {group.Key}");
                sb.AppendLine($"Count: {count}");
                sb.AppendLine($"Total Size: {FormatSize(totalSize)}");
                sb.AppendLine($"Average Size: {FormatSize(avgSize)}");

                // メモリ増加率の分析
                var growthAnalysis = AnalyzeGrowth(group);
                if (growthAnalysis.Any())
                {
                    sb.AppendLine("Growth Analysis:");
                    foreach (var analysis in growthAnalysis)
                    {
                        sb.AppendLine($"  {analysis}");
                    }
                }

                sb.AppendLine();
            }

            return sb.ToString();
        }

        private static bool IsObjectAlive(TrackedPythonObject obj)
        {
            return obj.Reference.TryGetTarget(out var target) && target != null && target.Handle != nint.Zero;
        }

        private static IEnumerable<string> AnalyzeGrowth(
            IGrouping<string, KeyValuePair<IntPtr, TrackedPythonObject>> group)
        {
            return group
                .Where(kvp => kvp.Value.SizeHistory.Count > 1)
                .Select(kvp =>
                {
                    var history = kvp.Value.SizeHistory;
                    var initialSize = history.First().Size;
                    var currentSize = history.Last().Size;
                    var growthFactor = (double)currentSize / initialSize;
                    var timeSpan = history.Last().Time - history.First().Time;

                    return $"Object grew from {FormatSize(initialSize)} to {FormatSize(currentSize)} " +
                           $"({growthFactor:F2}x) over {timeSpan.TotalSeconds:F1} seconds";
                });
        }

        private static string FormatSize(long bytes)
        {
            string[] suffixes = { "B", "KB", "MB", "GB", "TB" };
            int counter = 0;
            decimal size = bytes;

            while (size >= 1024 && counter < suffixes.Length - 1)
            {
                size /= 1024;
                counter++;
            }

            return $"{size:F2} {suffixes[counter]}";
        }
    }
}
