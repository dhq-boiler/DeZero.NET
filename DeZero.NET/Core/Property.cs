using System.ComponentModel;
using System.Runtime.CompilerServices;

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

        protected virtual void OnValueChanged(object value, [CallerMemberName] string? propertyName = null)
        {
            ValueChanged?.Invoke(this, new PropertyValueChangedEventArgs(propertyName, value));
            OnPropertyChanged(propertyName);
        }
    }

    public class Property<T> : Property
    {
        private readonly object _parent;
        private T _value;
        
        public Property()
        {
        }

        public Property(T value)
        {
            _value = value;
        }

        public T Value
        {
            get => _value;
            set
            {
                _value = value;
                OnValueChanged(value);
            }
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
