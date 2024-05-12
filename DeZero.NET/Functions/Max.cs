using DeZero.NET.Core;

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
            var y = x.Data.max(axis:Axis, keepdims:Keepdims);
            return [y.ToVariable()];
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Get<Variable>(0);
            var x = Inputs.ElementAt(0).Variable;
            var y = Outputs.ElementAt(0);

            var shape = Utils.max_backward_shape(x.Data, Axis);
            gy = gy.reshape(shape)[0];
            y = y.reshape(shape)[0];
            var cond = (x.Data == y.Data);
            gy = gy.Data.broadcast_to(cond.shape).ToVariable();
            return [gy * cond];
        }

        public static Variable[] Invoke(Variable x, int[] axis = null, bool keepdims = false)
        {
            return new Max(axis, keepdims).Call(Params.New.SetPositionalArgs(x));
        }
    }
}
