using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class Softmax : Function
    {
        public int[] Axis { get; }

        public Softmax(int[] axis = null)
        {
            if (axis is null)
            {
                Axis = [1];
            }
            else
            {
                Axis = axis;
            }
        }

        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>(0);
            using var y = x - x.Data.Value.max(axis: Axis, keepdims: true).ToVariable(this);
            var y2 = xp.exp(y.Data.Value).ToVariable(this);
            using var val = y2.Data.Value.sum(axis: new Axis(Axis), keepdims: true).ToVariable(this);
            y2 /= val;
            return [y2.Relay(this)];
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Get<Variable>(0);
            var y = Outputs.ElementAt(0);
            var gx = y * gy;
            using var sumdx = gx.Data.Value.sum(axis: new Axis(Axis), keepdims: true).ToVariable(this);
            gx -= y * sumdx;
            return [gx];
        }

        public static Variable[] Invoke(Variable x, int[] axis = null)
        {
            if (axis is null)
            {
                axis = [1];
            }
            return new Softmax(axis).Call(Params.New.SetPositionalArgs(x));
        }
    }
}
