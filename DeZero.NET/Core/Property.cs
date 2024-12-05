using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace DeZero.NET.Core
{
    public class Property : INotifyPropertyChanged, IDisposable
    {
        protected object _value;

        public event PropertyChangedEventHandler? PropertyChanged;

        public string PropertyName { get; set; }

        public object Value
        {
            get => _value;
            set
            {
                _value = value;
                OnPropertyChanged();
            }
        }

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

        public void Dispose()
        {
            if (Value is IDisposable disposable)
            {
                disposable.Dispose();
            }

            Value = default;
            GC.SuppressFinalize(this);
        }
    }

    [DebuggerDisplay("PropertyName={PropertyName}, Value={Value}")]
    public class Property<T> : Property, IDisposable
    {
        private readonly object _parent;
        
        public Property(string propertyName)
        {
            PropertyName = propertyName;
        }

        public Property(string propertyName, T value) : this(propertyName)
        {
            Value = value;
        }

        public Property(string propertyName, Action<T> valueChanged) : this(propertyName)
        {
            ValueChanged += (sender, e) => valueChanged((T)e.Value);
        }

        public T Value
        {
            get => (T)base.Value;
            set
            {
                base.Value = value;
                OnValueChanged(PropertyName, value);
            }
        }

        public void SetValueWithNoFireEvent(T value)
        {
            base.Value = value;
        }

        public void Dispose()
        {
            base.Dispose();
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
