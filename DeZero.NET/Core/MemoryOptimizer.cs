using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeZero.NET.Core
{
    public class MemoryOptimizer : IDisposable
    {
        private readonly List<IDisposable> _trackedResources = new();

        public void TrackResource(IDisposable resource)
        {
            _trackedResources.Add(resource);
        }

        public void Dispose()
        {
            foreach (var resource in _trackedResources)
            {
                resource?.Dispose();
            }
            _trackedResources.Clear();
        }
    }
}
