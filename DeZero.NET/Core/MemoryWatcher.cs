
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

        public bool IsMemoryIncreasing()
        {
            if (_usageHistory.Count < 5) return false;
            var trend = _usageHistory.Zip(_usageHistory.Skip(1), (a, b) => b - a);
            return trend.Average() > 2; // 平均2MB以上の増加傾向
        }

        public bool IsMemoryHigh()
        {
            if (_usageHistory.Count < 3)
                return false;

            var currentUsage = _usageHistory.Last();
            var averageUsage = _usageHistory.Average();

            // 以下の条件のいずれかを満たす場合にメモリ使用量が高いと判断
            return
                // 1. 現在の使用量が閾値を超えている
                currentUsage > _threshold ||

                // 2. 直近の平均使用量が閾値の80%を超えている
                averageUsage > _threshold * 0.8 ||

                // 3. 直近3回の測定で継続的に増加している
                _usageHistory.TakeLast(3)
                    .Zip(_usageHistory.TakeLast(3).Skip(1), (a, b) => b > a)
                    .All(x => x) &&
                // かつ、最新の使用量が閾値の60%を超えている
                currentUsage > _threshold * 0.6;
        }
    }
}
