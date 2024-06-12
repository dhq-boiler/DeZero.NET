using DeZero.NET.Layers;
using DeZero.NET.matplotlib;

namespace DeZero.NET.Models
{
    public abstract class Model : Layer
    {
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
                var _inputShape = currentOutput[0].Shape;
                currentOutput = layer.Call(currentOutput);
                var outputShape = currentOutput[0].Shape;

                Console.WriteLine($"{indentStr}  {layer.GetType().Name}:");
                Console.WriteLine($"{indentStr}    Input shape: {_inputShape.ToString()}");
                Console.WriteLine($"{indentStr}    Output shape: {outputShape.ToString()}");

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
