using Python.Runtime;
using System.Diagnostics;
using System.Text;

namespace DeZero.NET.Core
{
    public class MemoryCleanupScope : IDisposable
    {
        private readonly long _initialMemoryUsage;
        private readonly bool _isDebugMode;
        private readonly int _memoryThreshold;
        private readonly StringBuilder _log;

        public MemoryCleanupScope(int memoryThreshold = 100, bool isDebugMode = false)
        {
            _initialMemoryUsage = GpuMemoryMonitor.Instance.GetCurrentMemoryUsage();
            _memoryThreshold = memoryThreshold;
            _isDebugMode = isDebugMode;
            _log = new StringBuilder();

            if (_isDebugMode)
            {
                LogMemoryStatus("Cleanup Scope Started");
            }
        }

        public void ForceCleanup()
        {
            try
            {
                using (Py.GIL())
                {
                    // メモリプールの強制クリーンアップ
                    GpuMemoryMonitor.ForceMemoryPool();

                    // GILを解放した状態でGCを実行
                }

                // 未管理リソースの解放を促進
                GC.Collect(2, GCCollectionMode.Forced, true);

                // Pythonオブジェクトのファイナライズ中にGILが必要なため
                using (Py.GIL())
                {
                    var sw = Stopwatch.StartNew();
                    while (sw.ElapsedMilliseconds < 1000) // 1秒のタイムアウト
                    {
                        // GCが完了するまで短い間隔で待機
                        Thread.Sleep(10);
                        GC.WaitForPendingFinalizers();
                    }
                }

                // DeZero.NETのファイナライザを実行
                using (Py.GIL())
                {
                    Finalizer.Instance.Collect();
                }

                if (_isDebugMode)
                {
                    LogMemoryStatus("Forced Cleanup Completed");
                }
            }
            catch (Exception ex)
            {
                if (_isDebugMode)
                {
                    _log.AppendLine($"Cleanup error: {ex.Message}");
                }
            }
        }

        public void CheckAndCleanup()
        {
            var currentUsage = GpuMemoryMonitor.Instance.GetCurrentMemoryUsage();
            var delta = currentUsage - _initialMemoryUsage;

            if (_isDebugMode)
            {
                _log.AppendLine($"Memory delta: {delta}MB");
            }

            if (delta > _memoryThreshold || currentUsage > _memoryThreshold)
            {
                ForceCleanup();
            }
        }

        private void LogMemoryStatus(string message)
        {
            var currentUsage = GpuMemoryMonitor.Instance.GetCurrentMemoryUsage();
            _log.AppendLine($"{message} - Current Memory Usage: {currentUsage}MB");
        }

        public string GetLog()
        {
            return _log.ToString();
        }

        public void Dispose()
        {
            CheckAndCleanup();

            if (_isDebugMode)
            {
                var finalMemoryUsage = GpuMemoryMonitor.Instance.GetCurrentMemoryUsage();
                var totalDelta = finalMemoryUsage - _initialMemoryUsage;
                LogMemoryStatus($"Cleanup Scope Ended (Total Delta: {totalDelta}MB)");
                Console.WriteLine(_log.ToString());
            }
        }
    }

    public static class MemoryManagementExtensions
    {
        public static void SafeDispose(this IDisposable disposable)
        {
            try
            {
                disposable?.Dispose();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Dispose error: {ex.Message}");
            }
        }

        public static void SafeDisposeAll(this IEnumerable<IDisposable> disposables)
        {
            foreach (var disposable in disposables)
            {
                disposable.SafeDispose();
            }
        }
    }
}
