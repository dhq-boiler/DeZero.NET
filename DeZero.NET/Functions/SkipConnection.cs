using DeZero.NET.Core;
using DeZero.NET.Layers;

namespace DeZero.NET.Functions
{
    public class SkipConnection : Function
    {
        private Layer _layer;

        public SkipConnection(Layer layer)
        {
            _layer = layer;
        }

        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>(0);
            Variable h = _layer.Forward(x)[0];
            return [x + h];
        }

        public override Variable[] Backward(Params args)
        {
            var gout = args.Get<Variable>(0);

            if (_layer is Models.Sequential seq)
            {
                var gy = seq.Layers.First().Backward(gout);
                foreach (var layer in seq.Layers.Skip(1))
                {
                    gy = layer.Backward(gy);
                }

                return gy;
            }
            else
            {
                var gy = _layer.Backward(gout);
                return gy;
            }
        }

        public Variable[] Invoke(Variable x)
        {
            return Call(Params.New.SetPositionalArgs(x));
        }
    }
}
