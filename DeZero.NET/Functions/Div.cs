using DeZero.NET.Core;

namespace DeZero.NET.Functions
{
    public class Div : Function
    {
        public static Func<Params, Variable[]> F => x => [(x.Through.DistinctBy(x => x.Name).ElementAt(0).Variable.Data / x.Through.DistinctBy(x => x.Name).ElementAt(1).Variable.Data).ToVariable()];

        public Div()
        { }

        public Div(Func<Params, Variable[]> f)
            : base(f)
        { }

        public override Variable[] Forward(Params args)
        {
            var y = F(args)[0];
            return [y.Data.ToVariable(this)];
        }

        public override Variable[] Backward(Params args)
        {
            var gys = args.Get<Variable>(0);
            var (x0, x1) = (Inputs.ElementAt(0), Inputs.ElementAt(1));
            var gx0 = (gys.Data / x1.Variable.Data).ToVariable(this);
            var gx1 = (gys.Data * (-(x0.Variable.Data) / x1.Variable.Data.pow(2))).ToVariable(this);
            if (x0.Variable.Shape != x1.Variable.Shape)
            {
                gx0 = SumTo.Invoke(gx0, x0.Variable.Shape).Single();
                gx1 = SumTo.Invoke(gx1, x1.Variable.Shape).Single();
            }

            return [gx0, gx1];
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
