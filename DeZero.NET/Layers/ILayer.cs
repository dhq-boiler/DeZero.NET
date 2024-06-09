namespace DeZero.NET.Layers
{
    public interface ILayer
    {
        Variable[] Call(params Variable[] inputs);
    }
}
