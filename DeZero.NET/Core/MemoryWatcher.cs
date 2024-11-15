namespace DeZero.NET.Core
{
    public class MemoryWatcher
    {
        private readonly int _threshold;
        private int _previousUsage;
        private readonly Queue<int> _usageHistory = new(10);

        public MemoryWatcher(int threshold)
        {
            _threshold = threshold;
        }

        public void CheckAndCleanup(int iteration)
        {
            var currentUsage = (int)GpuMemoryMonitor.Instance.GetCurrentMemoryUsage();
            _usageHistory.Enqueue(currentUsage);
            if (_usageHistory.Count > 10) _usageHistory.Dequeue();

            if (IsMemoryIncreasing() || currentUsage > _threshold)
            {
                GpuMemoryMonitor.ForceMemoryPool();
                _usageHistory.Clear();
            }

            _previousUsage = currentUsage;
        }

        private bool IsMemoryIncreasing()
        {
            if (_usageHistory.Count < 5) return false;
            var trend = _usageHistory.Zip(_usageHistory.Skip(1), (a, b) => b - a);
            return trend.Average() > 2; // 平均2MB以上の増加傾向
        }
    }
}
