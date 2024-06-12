using DeZero.NET.Core;

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

            var y = x.reshape(new Shape(batchSize, channels, height * width))[0];
            y = y.Data.Value.sum(new Axis(2)).ToVariable();
            y = y / (height * width);

            return [y];
        }

        public override Variable[] Backward(Params input)
        {
            var gy = input.Get<Variable>(0);
            var x = Inputs.ElementAt(0).Variable;
            var batchSize = x.Shape[0];
            var channels = x.Shape[1];
            var height = x.Shape[2];
            var width = x.Shape[3];

            var gy2 = gy / (height * width);
            var gy3 = gy2.reshape(new Shape(batchSize, channels, 1, 1))[0];
            var gx = gy3.Data.Value.broadcast_to(new Shape(batchSize, channels, height, width))[0].ToVariable();

            return [gx];
        }
    }
}
