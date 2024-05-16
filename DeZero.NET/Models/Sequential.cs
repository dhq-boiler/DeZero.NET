using DeZero.NET.Layers;

namespace DeZero.NET.Models
{
    public abstract class Sequential : Model
    {
        public Layer[] Layers { get; }

        public Sequential(Layer[] layers) : base()
        {
            this.Layers = layers;
        }
        
        public override Variable[] Forward(params Variable[] x)
        {
            foreach (var layer in Layers)
            {
                x = layer.Forward(x);
            }
            return x;
        }
    }
}
