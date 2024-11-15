namespace DeZero.NET.Core
{
    public class BackwardMemoryManager : IDisposable
    {
        private readonly BatchScope _scope;
        private readonly int _cleanupInterval;
        private int _operationCount;

        public BackwardMemoryManager(int cleanupInterval = 5)
        {
            _scope = new BatchScope();
            _cleanupInterval = cleanupInterval;
            _operationCount = 0;
        }

        public void TrackTemporary(Variable var)
        {
            _scope.TrackTemporary(var);
            _operationCount++;

            if (_operationCount % _cleanupInterval == 0)
            {
                ForceCleanup();
            }
        }

        private void ForceCleanup()
        {
            GpuMemoryMonitor.ForceMemoryPool();
            GC.Collect();
        }

        public void Dispose()
        {
            _scope.Dispose();
            ForceCleanup();
        }
    }
}
