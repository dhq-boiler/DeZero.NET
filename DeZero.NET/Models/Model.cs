using DeZero.NET.Layers;
using DeZero.NET.Layers.Normalization;
using DeZero.NET.matplotlib;
using DeZero.NET.Optimizers;

namespace DeZero.NET.Models
{
    public abstract class Model : Layer
    {
        public Optimizer Optimizer { get; internal set; }

        public void Plot(Variable[] inputs, string to_file = "model.jpg")
        {
            var y = Forward(inputs)[0];
            pyplot.imshow(y.Data.Value);
            pyplot.show();
            //Utils.plot_dot_graph(y, verbose: true, to_file: to_file);
        }

        public virtual void Explain(Shape inputShape, int indent = 0)
        {
            //var sb = new StringBuilder();
            var indentStr = new string(' ', indent);

            Console.WriteLine($"{indentStr}Model:");
            var currentOutput = new[] { new Variable(xp.zeros(inputShape), name: "input") };
            foreach (var layer in EnumerateLayers())
            {
                Shape outputShape = default;
                using (var _ = new UsingConfig("EnableBackprop", false))
                {
                    if (layer is BatchNorm bn)
                    {
                        //bn.InitParams(currentOutput[0]);
                    }

                    var _inputShape = currentOutput[0].Shape;
                    currentOutput = layer.Call(currentOutput);
                    outputShape = currentOutput[0].Shape;

                    Console.WriteLine($"{indentStr}  {layer.GetType().Name}:");
                    Console.WriteLine($"{indentStr}    Input shape: {_inputShape.ToString()}");
                    if (layer is IWbOwner owner)
                    {
                        Console.WriteLine($"{indentStr}    W.shape={owner.W.Value.Shape.ToString()}");
                        Console.WriteLine($"{indentStr}    b.shape={owner.b.Value.Shape.ToString()}");
                    }

                    Console.WriteLine($"{indentStr}    Output shape: {outputShape.ToString()}");
                }

                if (layer is Model subModel)
                {
                    subModel.Explain(outputShape, indent + 4);
                }
            }
        }

        protected virtual IEnumerable<Layer> EnumerateLayers()
        {
            yield break;
        }
    }
}
