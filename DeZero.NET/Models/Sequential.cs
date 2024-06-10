using DeZero.NET.Layers;

namespace DeZero.NET.Models
{
    public class Sequential : Model
    {
        public List<Layer> Layers { get; }

        public Sequential(Layer[] layers) : base()
        {

            Layers = new List<Layer>();

            for (int i = 0; i < layers.Length; i++)
            {
                var layer = layers[i];
                SetAttribute($"l{i}", layer);
                Layers.Add(layer);
            }
        }
        
        public override Variable[] Forward(params Variable[] x)
        {
            var index = 0;
            foreach (var layer in Layers)
            {
                x = layer.Call(x);
                index++;
            }
            return x;
        }

        public void DisposeAllInputs()
        {
            foreach (var layer in Layers)
            {
                layer.DisposeAllInputs();
            }
        }
    }
}
