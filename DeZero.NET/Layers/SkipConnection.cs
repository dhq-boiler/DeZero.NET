using DeZero.NET.Models;

namespace DeZero.NET.Layers
{
    public class SkipConnection : Model
    {
        public Layer Layer { get; }

        public SkipConnection(Layer layer)
        {
            this.Layer = layer;
        }

        public override Variable[] Forward(params Variable[] xs)
        {
            return new Functions.SkipConnection(this.Layer).Invoke(xs[0]);
        }

        protected override System.Collections.Generic.IEnumerable<Layer> EnumerateLayers()
        {
            yield return this.Layer;
        }
    }
}
