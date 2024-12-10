using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class Sqrt : Function
    {
        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>(0);
            var y = x.Data.Value.sqrt();
            return [y.Relay(this)];
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Get<Variable>(0);
            var x = Inputs.ElementAt(0).Variable;
            var y = Outputs.ElementAt(0);

            // 数値安定性のためのイプシロン
            const float eps = 1e-7f;

            // ∂sqrt(x)/∂x = 1/(2√x) = 1/(2y)
            using var yeps = y.Data.Value + eps;
            using var yesp05 = 0.5f / yeps;
            using var gx = gy.Data.Value * yesp05;

            return [gx.copy().ToVariable(this)];
        }

        public static Variable[] Invoke(Variable x)
        {
            return new Sqrt().Call(Params.New.SetPositionalArgs(x));
        }
    }
}
