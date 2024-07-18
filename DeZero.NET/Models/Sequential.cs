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
            for (int i = 0; i < Layers.Count; i++)
            {
                var layer = Layers[i];
                x = layer.Call(x);
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

        protected override IEnumerable<Layer> EnumerateLayers()
        {
            foreach (var layer in Layers)
            {
                yield return layer;
            }
        }
    }
}
