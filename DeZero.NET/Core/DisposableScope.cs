namespace DeZero.NET.Core
{
    public class DisposableScope : IDisposable
    {
        private readonly HashSet<IDisposable> _trackedObjects = new();

        public void Track(IDisposable obj)
        {
            if (obj != null) _trackedObjects.Add(obj);
        }

        public void Dispose()
        {
            foreach (var obj in _trackedObjects)
            {
                obj?.Dispose();
            }
            _trackedObjects.Clear();
        }
    }
}
