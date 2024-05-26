using DeZero.NET.Layers;

namespace DeZero.NET.Models
{
    public class MLP : Model
    {
        public Function Activation { get; set; }
        public List<Layer> Layers { get; }

        public MLP(int[] fc_output_sizes, Function activation) : base()
        {
            if (activation is null)
            {
                activation = new Functions.Sigmoid();
            }
            else
            {
                Activation = activation;
            }

            Layers = new List<Layer>();

            for (int i = 0; i < fc_output_sizes.Length; i++)
            {
                var out_size = fc_output_sizes[i];
                var layer = new Layers.Linear(out_size);
                SetAttribute($"l{i}", layer);
                Layers.Add(layer);
            }
        }

        public override Variable[] Forward(params Variable[] xs)
        {
            Variable x = xs[0];
            for (int i = 0; i < Layers.Count - 1; i++)
            {
                var layer = Layers[i];
                var layerOut = layer.Call(x)[0];
                x = Activation.Call(Core.Params.New.SetPositionalArgs(layerOut, arg1Name: "x"))[0];
            }

            return Layers.Last().Call(x);
        }
    }
}
