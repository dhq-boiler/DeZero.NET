using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeZero.NET.Core
{
    internal sealed class ComputeMemoryScope : IDisposable
    {
        private readonly Variable[] _variables;
        private bool _disposed;

        public ComputeMemoryScope(params Variable[] variables)
        {
            _variables = variables;
        }

        public void Dispose()
        {
            if (_disposed) return;
            foreach (var v in _variables) v?.Dispose();
            _disposed = true;
        }
    }
}
