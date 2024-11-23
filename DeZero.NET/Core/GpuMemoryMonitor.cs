using DeZero.NET.Log;
using Python.Runtime;
using System.Collections.Concurrent;
using System.Diagnostics;

namespace DeZero.NET.Core
{
    public class GpuMemoryMonitor : IDisposable
    {
        private static readonly Lazy<GpuMemoryMonitor> _instance = new(() => new GpuMemoryMonitor());
        public static LogLevel LogLevel { get; set; } = LogLevel.Info;
        public static bool IsVerbose { get; set; } = false;

        // メモリ閾値の定数
        private const double WARNING_THRESHOLD = 0.85;
        private const int MEMORY_DELTA_THRESHOLD = 100;
        private const int LOG_RETENTION_HOURS = 24;
        private const int LOG_CLEANUP_INTERVAL = 1;
        private const int CHECKPOINT_CLEANUP_INTERVAL = 1000;

        public static GpuMemoryMonitor Instance => _instance.Value;

        private readonly ILogger _logger;
        private readonly ConcurrentDictionary<string, (long Timestamp, long MemoryUsed)> _checkpoints;
        private readonly string _logFilePath;
        private readonly object _logLock = new();
        private readonly Timer _cleanupTimer;
        private volatile bool _isDisposed;

        // キャッシュされたPythonオブジェクト
        private dynamic _cupy;
        private dynamic _mempool;
        private int _checkpointCount;

        private GpuMemoryMonitor()
        {
            _logger = new ConsoleLogger(LogLevel, isVerbose: IsVerbose);
            _checkpoints = new ConcurrentDictionary<string, (long, long)>();
            _logFilePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "gpu_memory_log.txt");

            InitializeLog();
            InitializePythonObjects();

            // 定期的なクリーンアップタイマーを設定
            _cleanupTimer = new Timer(CleanupOldData, null, TimeSpan.FromHours(1), TimeSpan.FromHours(1));
        }

        private void InitializePythonObjects()
        {
            try
            {
                using (Py.GIL())
                {
                    _cupy = Py.Import("cupy");
                    _mempool = _cupy.get_default_memory_pool();
                }
            }
            catch (Exception ex)
            {
                _logger.LogError($"Failed to initialize Python objects: {ex.Message}");
            }
        }

        private void InitializeLog()
        {
            try
            {
                var logDir = Path.GetDirectoryName(_logFilePath);
                if (!Directory.Exists(logDir))
                {
                    Directory.CreateDirectory(logDir);
                }

                if (!File.Exists(_logFilePath))
                {
                    var header = "Timestamp,Location,Total Memory (MB),Used Memory (MB),Free Memory (MB)";
                    File.WriteAllText(_logFilePath, header + Environment.NewLine);
                }

                // 古いログファイルのクリーンアップ
                CleanupOldLogFiles();
            }
            catch (Exception ex)
            {
                _logger.LogError($"Failed to initialize log: {ex.Message}");
            }
        }

        private void CleanupOldLogFiles()
        {
            try
            {
                var logDir = Path.GetDirectoryName(_logFilePath);
                var oldFiles = Directory.GetFiles(logDir, "gpu_memory_log*.txt")
                    .Where(f => File.GetCreationTime(f) < DateTime.Now.AddHours(-LOG_RETENTION_HOURS));

                foreach (var file in oldFiles)
                {
                    try
                    {
                        File.Delete(file);
                    }
                    catch (Exception ex)
                    {
                        _logger.LogWarning($"Failed to delete old log file {file}: {ex.Message}");
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError($"Failed to cleanup old log files: {ex.Message}");
            }
        }

        private void CleanupOldData(object state)
        {
            if (_isDisposed) return;

            try
            {
                var cutoffTime = DateTimeOffset.UtcNow.AddMinutes(-LOG_CLEANUP_INTERVAL).ToUnixTimeMilliseconds();
                var keysToRemove = _checkpoints
                    .Where(kvp => kvp.Value.Timestamp < cutoffTime)
                    .Select(kvp => kvp.Key)
                    .ToList();

                foreach (var key in keysToRemove)
                {
                    _checkpoints.TryRemove(key, out _);
                }

                CleanupOldLogFiles();
            }
            catch (Exception ex)
            {
                _logger.LogError($"Failed to cleanup old data: {ex.Message}");
            }
        }

        public static void ForceMemoryPool()
        {
            try
            {
                using (Py.GIL())
                {
                    Instance._mempool?.free_all_blocks();
                    Instance._cupy?.get_default_pinned_memory_pool()?.free_all_blocks();
                }
            }
            catch (Exception ex)
            {
                Instance._logger.LogError($"Failed to force memory pool: {ex.Message}");
            }
        }

        public long GetCurrentMemoryUsage()
        {
            try
            {
                using (Py.GIL())
                {
                    return Parse(Instance._mempool.used_bytes()) / (1024 * 1024);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError($"Failed to get current memory usage: {ex.Message}");
                return 0;
            }
        }

        private long Parse(PyObject pyObject)
        {
            try
            {
                return pyObject?.ToString() is string str ? long.Parse(str) : 0;
            }
            catch
            {
                return 0;
            }
        }

        public void LogMemoryUsage(string location, bool verbose = false)
        {
            if (_isDisposed) return;

            verbose |= IsVerbose;

            try
            {
                using (Py.GIL())
                {
                    if (_mempool == null)
                    {
                        InitializePythonObjects();
                    }

                    var (totalMemory, usedMemory, freeMemory) = GetMemoryInfo();
                    var timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();

                    LogToFile(timestamp, location, totalMemory, usedMemory, freeMemory);
                    CheckAndLogMemoryDelta(location, timestamp, usedMemory, totalMemory, verbose);

                    // チェックポイントの数を監視し、必要に応じてクリーンアップ
                    if (Interlocked.Increment(ref _checkpointCount) % CHECKPOINT_CLEANUP_INTERVAL == 0)
                    {
                        CleanupOldData(null);
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError($"Failed to log memory usage: {ex.Message}");
            }
        }

        private (long Total, long Used, long Free) GetMemoryInfo()
        {
            var totalMemory = Parse(_mempool.total_bytes()) / (1024 * 1024);
            var usedMemory = Parse(_mempool.used_bytes()) / (1024 * 1024);
            return (totalMemory, usedMemory, totalMemory - usedMemory);
        }

        private void LogToFile(long timestamp, string location, long totalMemory, long usedMemory, long freeMemory)
        {
            try
            {
                var logEntry = $"{DateTime.Now:yyyy-MM-dd HH:mm:ss.fff},{location},{totalMemory},{usedMemory},{freeMemory}";
                lock (_logLock)
                {
                    File.AppendAllText(_logFilePath, logEntry + Environment.NewLine);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError($"Failed to write to log file: {ex.Message}");
            }
        }

        private void CheckAndLogMemoryDelta(string location, long timestamp, long usedMemory, long totalMemory, bool verbose)
        {
            var previousCheckpoint = _checkpoints.GetOrAdd(location, (timestamp, usedMemory));
            var memoryDelta = usedMemory - previousCheckpoint.MemoryUsed;
            _checkpoints.TryUpdate(location, (timestamp, usedMemory), previousCheckpoint);

            if (verbose || Math.Abs(memoryDelta) > MEMORY_DELTA_THRESHOLD)
            {
                LogMemoryStats(location, totalMemory, usedMemory, memoryDelta);
            }

            if ((double)usedMemory / totalMemory > WARNING_THRESHOLD)
            {
                HandleHighMemoryUsage(location, usedMemory, totalMemory, verbose);
            }
        }

        private void LogMemoryStats(string location, long totalMemory, long usedMemory, long memoryDelta)
        {
            _logger.LogDebug($"""
                Location: {location}
                Total Memory: {totalMemory:N0} MB
                Used Memory: {usedMemory:N0} MB
                Free Memory: {(totalMemory - usedMemory):N0} MB
                Memory Change: {(memoryDelta >= 0 ? "+" : "")}{memoryDelta:N0} MB
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

        public void ClearCheckpoint(string location)
        {
            _checkpoints.TryRemove(location, out _);
            Interlocked.Decrement(ref _checkpointCount);
        }

        public void Dispose()
        {
            if (_isDisposed) return;

            _isDisposed = true;
            _cleanupTimer?.Dispose();
            _checkpoints.Clear();

            try
            {
                using (Py.GIL())
                {
                    _mempool = null;
                    _cupy = null;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError($"Error during disposal: {ex.Message}");
            }
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
