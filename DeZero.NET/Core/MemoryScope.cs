namespace DeZero.NET.Core
{
    public class MemoryScope : IDisposable
    {
        private readonly List<IDisposable> _resources = new();
        private readonly List<IDisposable> _outputResources = new();
        private bool _isDisposed;

        public T Register<T>(T resource) where T : IDisposable
        {
            if (_isDisposed)
                throw new ObjectDisposedException(nameof(MemoryScope));

            _resources.Add(resource);
            return resource;
        }

        public T RegisterForOutput<T>(T resource) where T : IDisposable
        {
            if (_isDisposed)
                throw new ObjectDisposedException(nameof(MemoryScope));

            _outputResources.Add(resource);
            return resource;
        }

        public void Dispose()
        {
            if (_isDisposed) return;

            // 中間リソースを即座に解放
            foreach (var resource in _resources)
            {
                try { resource?.Dispose(); }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error disposing resource: {ex.Message}");
                }
            }
            _resources.Clear();

            // 出力リソースは保持（呼び出し元で管理）
            _outputResources.Clear();

            _isDisposed = true;
        }
    }
}
