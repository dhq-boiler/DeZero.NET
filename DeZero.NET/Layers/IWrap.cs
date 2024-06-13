using DeZero.NET.Core;

namespace DeZero.NET.Layers
{
    public interface IWrap : ILayer
    {
        Property<IWbOwner> Layer { get; }
    }
}
