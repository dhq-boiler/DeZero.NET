namespace DeZero.NET.Core
{
    public class BatchScope : IDisposable
    {
        private readonly HashSet<Variable> _temporaryVariables = new();
        private readonly HashSet<Variable> _preservedVariables = new();
        private bool _isDisposed;

        public void TrackTemporary(Variable var)
        {
            if (var != null && !_preservedVariables.Contains(var) && !_temporaryVariables.Contains(var))
            {
                _temporaryVariables.Add(var);
            }
        }

        public void PreserveVariable(Variable var)
        {
            if (var != null)
            {
                _preservedVariables.Add(var);
                _temporaryVariables.Remove(var);
            }
        }

        public void Dispose()
        {
            if (_isDisposed) return;

            foreach (var var in _temporaryVariables)
            {
                if (var != null && !_preservedVariables.Contains(var))
                {
                    var.Dispose();
                }
            }

            _temporaryVariables.Clear();
            _isDisposed = true;
        }

        ~BatchScope()
        {
            Dispose();
        }
    }
}