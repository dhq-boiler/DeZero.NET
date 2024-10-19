using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class Sum : Function
    {
        public Axis Axis { get; }
        public bool? Keepdims { get; }
        public Shape x_shape { get; private set; }

        public Sum(Axis axis, bool? keepdims = null)
        {
            Axis = axis;
            Keepdims = keepdims;
        }

        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>(0);
            x_shape = x.Shape;
            var y = xp.sum(x.Data.Value, axis: this.Axis, keepdims: Keepdims);
            return [y.ToVariable(this)];
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Get<Variable>(0);
            gy = Utils.reshape_sum_backward(gy, x_shape, Axis, Keepdims);
            var gx = BroadcastTo.Invoke(gy, x_shape);
            return gx;
        }

        public static Variable[] Invoke(Variable x, Axis axis = null, bool keepdims = false)
        {
            return new Sum(axis, keepdims).Call(Params.New.SetPositionalArgs(x));
        }
    }
}
