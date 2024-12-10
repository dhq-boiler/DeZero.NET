using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class Where : Function
    {
        private Variable _condition;
        private Variable _x;
        private Variable _y;

        public override Variable[] Forward(Params args)
        {
            _condition = args.Get<Variable>(0);
            _x = args.Get<Variable>(1);
            _y = args.Get<Variable>(2);

            var result = xp.where(_condition.Data.Value, _x.Data.Value, _y.Data.Value);
            return [result.Relay(this)];
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Through[0].Variable;
            using var gx = DeZero.NET.Functions.Mul.Invoke(_condition, gy)[0];
            using var one = xp.array(1).ToVariable();
            using var sub = DeZero.NET.Functions.Sub.Invoke(one, _condition)[0];
            using var gy_inv = DeZero.NET.Functions.Mul.Invoke(sub, gy)[0];
            using var gcondition = xp.zeros_like(_condition.Data.Value).ToVariable();
            return [gcondition.copy(), gx.copy(), gy_inv.copy()];
        }

        public static (Variable[], Function) Invoke(Variable condition, Variable x, Variable y)
        {
            var f = new Where();
            var output = f.Call(Params.New.SetPositionalArgs(condition, x, y));
            return (output, f);
        }
    }
}
