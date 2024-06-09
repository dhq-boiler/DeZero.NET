using DeZero.NET.Core;

namespace DeZero.NET.Layers
{
    public interface IWeight : ILayer
    {
        Property<Parameter> W { get; }
        Action WInitialized { get; set; }
    }
}
