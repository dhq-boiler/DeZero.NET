using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class Mul : Function
    {
        public Func<Params, Variable[]> F => x => [(x.Get<Variable>(0).Data.Value * x.Get<Variable>(1).Data.Value).ToVariable(this)];
        public Shape X0_Shape { get; set; }
        public Shape X1_Shape { get; set; }

        public Mul()
        { }

        public Mul(Func<Params, Variable[]> f)
            : base(f)
        { }

        public override Variable[] Forward(Params args)
        {
            var xs = args.Through;
            var x0 = xs[0].Variable;
            var x1 = xs[1].Variable;
            X0_Shape = x0.Shape;
            X1_Shape = x1.Shape;
            var y = F(Params.New.SetPositionalArgs(x0, x1))[0];
            return [y.Relay(this, x0, x1).copy()];
        }

        public override Variable[] Backward(Params args)
        {
            var gys = args.Through;
            var x0 = this.Inputs.ElementAt(0).Variable;
            var x1 = this.Inputs.ElementAt(1).Variable;
            using var _gx0 = gys.Single().Variable.Data.Value * x1.Data.Value;
            using var _gx1 = gys.Single().Variable.Data.Value * x0.Data.Value;
            using var gx0 = _gx0.ToVariable();
            using var gx1 = _gx1.ToVariable();
            if (X0_Shape != X1_Shape)
            {
                using var __gx0 = SumTo.Invoke(gx0, X0_Shape).Single();
                gx0.Dispose();
                using var __gx1 = SumTo.Invoke(gx1, X1_Shape).Single();
                gx1.Dispose();
                return [__gx0.copy(), __gx1.copy()];
            }

            return [gx0.copy(), gx1.copy()];
        }

        public static Variable[] Invoke(Variable x0, Variable x1)
        {
            return new Mul().Call(Params.New.SetPositionalArgs(x0, x1));
        }
    }
}
