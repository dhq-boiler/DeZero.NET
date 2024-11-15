namespace DeZero.NET.Core
{
    using Python.Runtime;
    using System.Collections.Concurrent;
    using System.Diagnostics;

    public class GpuMemoryMonitor : IDisposable
    {
        private static readonly Lazy<GpuMemoryMonitor> _instance = new(() => new GpuMemoryMonitor());
        public static GpuMemoryMonitor Instance => _instance.Value;

        private readonly ConcurrentDictionary<string, (long Timestamp, long MemoryUsed)> _checkpoints;
        private readonly string _logFilePath;
        private readonly object _logLock = new();
        private bool _isDisposed;

        private GpuMemoryMonitor()
        {
            _checkpoints = new ConcurrentDictionary<string, (long, long)>();
            _logFilePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "gpu_memory_log.txt");
            InitializeLog();
        }

        private void InitializeLog()
        {
            var header = $"Timestamp,Location,Total Memory (MB),Used Memory (MB),Free Memory (MB)";
            File.WriteAllText(_logFilePath, header + Environment.NewLine);
        }

        private long Parse(PyObject pyObject)
        {
            return long.Parse(pyObject.ToString());
        }

        public static void ForceMemoryPool()
        {
            using (Py.GIL())
            {
                // CuPyのメモリプールを強制的に解放
                dynamic cupy = Py.Import("cupy");
                dynamic mempool = cupy.get_default_memory_pool();
                mempool.free_all_blocks();

                // 必要に応じてpinnedメモリプールも解放
                dynamic pinned_mempool = cupy.get_default_pinned_memory_pool();
                pinned_mempool.free_all_blocks();
            }
        }

        public long GetCurrentMemoryUsage()
        {
            try
            {
                using (Py.GIL())
                {
                    dynamic cupy = Py.Import("cupy");
                    dynamic mempool = cupy.get_default_memory_pool();
                    return Parse(mempool.used_bytes()) / (1024 * 1024); // Convert to MB
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error getting GPU memory usage: {ex.Message}");
                return 0;
            }
        }

        public void LogMemoryUsage(string location, bool verbose = false)
        {
            if (_isDisposed) return;

            try
            {
                using (Py.GIL())
                {
                    // CuPyのメモリ情報を取得
                    dynamic cupy = Py.Import("cupy");
                    dynamic mempool = cupy.get_default_memory_pool();
                    //dynamic pinned_mempool = cupy.get_default_pinned_memory_pool();

                    // 合計・使用中・空きメモリを計算 (バイトからMBに変換)
                    long totalMemory = Parse(mempool.total_bytes()) / (1024 * 1024);
                    long usedMemory = Parse(mempool.used_bytes()) / (1024 * 1024);
                    long freeMemory = totalMemory - usedMemory;

                    var timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
                    var logEntry = $"{DateTime.Now:yyyy-MM-dd HH:mm:ss.fff},{location},{totalMemory},{usedMemory},{freeMemory}";

                    lock (_logLock)
                    {
                        File.AppendAllText(_logFilePath, logEntry + Environment.NewLine);
                    }

                    // メモリ使用量の変化を記録
                    var previousCheckpoint = _checkpoints.GetOrAdd(location, (timestamp, usedMemory));
                    var memoryDelta = usedMemory - previousCheckpoint.MemoryUsed;
                    _checkpoints[location] = (timestamp, usedMemory);

                    if (verbose || Math.Abs(memoryDelta) > 100) // 100MB以上の変化があった場合に出力
                    {
                        Console.WriteLine($"""
                        Location: {location}
                        Total Memory: {totalMemory:N0} MB
                        Used Memory: {usedMemory:N0} MB
                        Free Memory: {freeMemory:N0} MB
                        Memory Change: {(memoryDelta >= 0 ? "+" : "")}{memoryDelta:N0} MB
                        """);
                    }

                    // メモリ使用量が危険水準に達した場合の警告
                    const double warningThreshold = 0.85; // 85%
                    if ((double)usedMemory / totalMemory > warningThreshold)
                    {
                        Console.WriteLine($"WARNING: High GPU memory usage detected at {location}! ({usedMemory:N0}/{totalMemory:N0} MB)");
                        GC.Collect();
                        Finalizer.Instance.Collect();
                        ForceMemoryPool();
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error logging GPU memory usage: {ex.Message}");
            }
        }

        public void ClearCheckpoint(string location)
        {
            _checkpoints.TryRemove(location, out _);
        }

        public void Dispose()
        {
            if (_isDisposed) return;
            _isDisposed = true;
            _checkpoints.Clear();
        }
    }

    // メモリ使用量の追跡を容易にするためのコンテキストマネージャー
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

    // 使用例を示すための拡張メソッド
    public static class MemoryMonitoringExtensions
    {
        public static IDisposable TrackMemory(this object _, string location, bool verbose = false)
        {
            return new MemoryTrackingScope(location, verbose);
        }
    }
}
