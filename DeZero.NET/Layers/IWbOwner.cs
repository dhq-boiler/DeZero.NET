using DeZero.NET.Core;

namespace DeZero.NET.Layers
{
    public interface IWbOwner : ILayer
    {
        Property<Parameter> W { get; }
        Property<Parameter> b { get; }
        Action WInitialized { get; set; }
    }
}
