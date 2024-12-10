using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class Div : Function
    {
        public static Func<Params, Variable[]> F => x => [(x.Through.DistinctBy(x => x.Name).ElementAt(0).Variable.Data.Value / x.Through.DistinctBy(x => x.Name).ElementAt(1).Variable.Data.Value).ToVariable()];

        public Div()
        { }

        public Div(Func<Params, Variable[]> f)
            : base(f)
        { }

        public override Variable[] Forward(Params args)
        {
            using var y = F(args)[0];
            return [y.Relay(this, [args.Through.DistinctBy(x => x.Name).ElementAt(0).Variable, args.Through.DistinctBy(x => x.Name).ElementAt(1).Variable]).copy()];
        }

        public override Variable[] Backward(Params args)
        {
            var gys = args.Get<Variable>(0);
            var (x0, x1) = (Inputs.ElementAt(0), Inputs.ElementAt(1));
            using var d = gys.Data.Value / x1.Variable.Data.Value;
            using var gx0 = d.ToVariable(this);
            using var a = -x0.Variable.Data.Value;
            using var b = a / x1.Variable.Data.Value.pow(2);
            using var c = gys.Data.Value * b;
            using var gx1 = c.ToVariable(this);
            if (x0.Variable.Shape != x1.Variable.Shape)
            {
                using var _gx0 = SumTo.Invoke(gx0, x0.Variable.Shape).Single();
                using var _gx1 = SumTo.Invoke(gx1, x1.Variable.Shape).Single();
                return [_gx0.copy(), _gx1.copy()];
            }

            return [gx0.copy(), gx1.copy()];
        }

        public static Variable[] Invoke(Variable x0, Variable x1)
        {
            return new Div().Call(Params.New.SetPositionalArgs(x0, x1));
        }

        public static Variable[] ReverseInvoke(Variable x0, Variable x1)
        {
            return new Div().Call(Params.New.SetPositionalArgs(x1, x0));
        }
    }
}
