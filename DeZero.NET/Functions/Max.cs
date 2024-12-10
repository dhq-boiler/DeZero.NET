using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class Max : Function
    {
        public int[] Axis { get; }
        public bool Keepdims { get; }

        public Max(int[] axis = null, bool keepdims = false)
        {
            Axis = axis;
            Keepdims = keepdims;
        }

        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>(0);
            using var y = x.Data.Value.max(axis:Axis, keepdims:Keepdims);
            return [y.copy().Relay(this)];
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Get<Variable>(0);
            var x = Inputs.ElementAt(0).Variable;
            var y = Outputs.ElementAt(0);

            using var shape = Utils.max_backward_shape(x.Data.Value, Axis);
            using var _gy = gy.reshape(shape)[0];
            using var _y = y.reshape(shape)[0];
            using var cond = (x.Data.Value == _y.Data.Value);
            using var cond_shape = cond.shape;
            using var __gy = _gy.Data.Value.broadcast_to(cond_shape).ToVariable();
            using var result = __gy * cond;
            return [result.copy()];
        }

        public static Variable[] Invoke(Variable x, int[] axis = null, bool keepdims = false)
        {
            return new Max(axis, keepdims).Call(Params.New.SetPositionalArgs(x));
        }
    }
}
