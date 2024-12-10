using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class GlobalAveragePooling : Function
    {
        public override Variable[] Forward(Params input)
        {
            var x = input.Get<Variable>(0);
            var batchSize = x.Shape[0];
            var channels = x.Shape[1];
            var height = x.Shape[2];
            var width = x.Shape[3];

            using var y = x.reshape(new Shape(batchSize, channels, height * width))[0];
            using var _y = y.Data.Value.sum(new Axis(2)).ToVariable();
            using var __y = _y / (height * width);

            return [__y.copy().Relay(this)];
        }

        public override Variable[] Backward(Params input)
        {
            var gy = input.Get<Variable>(0);
            var x = Inputs.ElementAt(0).Variable;
            var batchSize = x.Shape[0];
            var channels = x.Shape[1];
            var height = x.Shape[2];
            var width = x.Shape[3];

            using var gy2 = gy / (height * width);
            using var gy3 = gy2.reshape(new Shape(batchSize, channels, 1, 1))[0];
            using var gx = gy3.Data.Value.broadcast_to(new Shape(batchSize, channels, height, width))[0].ToVariable();

            return [gx.copy()];
        }

        public static Variable[] Invoke(Variable x)
        {
            var op = new GlobalAveragePooling();
            return op.Call(Params.New.SetPositionalArgs(x));
        }
    }
}
