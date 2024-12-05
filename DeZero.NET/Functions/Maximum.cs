using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class Maximum : Function
    {
        private Shape _x0_shape;
        private Shape _x1_shape;
        private Variable _x0;
        private Variable _x1;

        public override Variable[] Forward(Params args)
        {
            _x0 = args.Get<Variable>(0).copy();
            _x1 = args.Get<Variable>(1).copy();

            // Store original shapes for backward pass
            _x0_shape = _x0.Shape;
            _x1_shape = _x1.Shape;

            var y = xp.maximum(_x0.Data.Value, _x1.Data.Value);
            return [y.Relay(this)];
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Through[0].Variable;

            // Compare using original tensors to get correct broadcasting
            var condition = GreaterThanOrEqual.Invoke(_x0, _x1).Item1[0];

            // Create gradients using where operation
            var gx0 = Where.Invoke(
                condition,
                gy,
                xp.zeros_like(gy.Data.Value).ToVariable()
            ).Item1[0];

            var gx1 = Where.Invoke(
                DeZero.NET.Functions.Not.Invoke(condition).Item1[0],
                gy,
                xp.zeros_like(gy.Data.Value).ToVariable()
            ).Item1[0];

            // 勾配を元の入力形状に合わせる
            gx0 = BroadcastUtils.SumToShape(gx0, _x0_shape);
            gx1 = BroadcastUtils.SumToShape(gx1, _x1_shape);

            return new[] { gx0, gx1 };
        }

        public static (Variable[], Function) Invoke(Variable x0, Variable x1)
        {
            var f = new Maximum();
            var y = f.Call(Params.New.SetPositionalArgs(x0, x1));
            return (y, f);
        }
    }
}
