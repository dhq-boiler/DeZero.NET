using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using Cupy;

namespace DeZero.NET.Core
{
    public class Property : INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler? PropertyChanged;

        protected virtual void OnPropertyChanged([CallerMemberName] string? propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        protected bool SetField<T>(ref T field, T value, [CallerMemberName] string? propertyName = null)
        {
            if (EqualityComparer<T>.Default.Equals(field, value)) return false;
            field = value;
            OnPropertyChanged(propertyName);
            return true;
        }

        public delegate void PropertyValueChangedEventHandler(object sender, PropertyValueChangedEventArgs e);

        public event PropertyValueChangedEventHandler? ValueChanged;

        protected virtual void OnValueChanged(string? propertyName, object value)
        {
            ValueChanged?.Invoke(this, new PropertyValueChangedEventArgs(propertyName, value));
            OnPropertyChanged(propertyName);
        }
    }

    [DebuggerDisplay("PropertyName={PropertyName}, Value={Value}")]
    public class Property<T> : Property, IDisposable
    {
        public string PropertyName { get; }
        private readonly object _parent;
        private T _value;
        
        public Property(string propertyName)
        {
            PropertyName = propertyName;
        }

        public Property(string propertyName, T value) : this(propertyName)
        {
            _value = value;
        }

        public T Value
        {
            get => _value;
            set
            {
                _value = value;
                OnValueChanged(PropertyName, value);
            }
        }

        public void Dispose()
        {
            if (this._value is IDisposable disposable)
            {
                disposable.Dispose();
            }

            this._value = default;
        }
    }

    public class PropertyValueChangedEventArgs : PropertyChangedEventArgs
    {
        public object Value { get; }

        public PropertyValueChangedEventArgs(string propertyName, object value) : base(propertyName)
        {
            Value = value;
        }
    }
}
