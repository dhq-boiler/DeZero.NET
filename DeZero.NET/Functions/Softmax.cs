using DeZero.NET.Core;

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
            var x = args.Get<Variable>("x");
            var y = x - x.Data.max(axis: Axis, keepdims: true).ToVariable(this);
            y = xp.exp(y.Data).ToVariable(this);
            y /= y.Data.sum(axis: new Axis(Axis), keepdims: true).ToVariable(this);
            return [y];
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Get<Variable>(0);
            var y = Outputs.ElementAt(0);
            var gx = y * gy;
            var sumdx = gx.Data.sum(axis: new Axis(Axis), keepdims: true).ToVariable(this);
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
