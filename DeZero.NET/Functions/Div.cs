using DeZero.NET.Core;

namespace DeZero.NET.Functions
{
    public class Div : Function
    {
        public static Func<Params, Variable[]> F => x => [(x.Get<Variable>("x0").Data / x.Get<Variable>("x1").Data).ToVariable()];

        public Div()
        { }

        public Div(Func<Params, Variable[]> f)
            : base(f)
        { }

        public override Variable[] Forward(Params args)
        {
            var xs = args.Through();
            var x0 = xs[0];
            var x1 = xs[1];
            var y = F(Params<Variable, Variable>.args(x0, x1))[0];
            return [y];
        }

        public override Variable[] Backward(Params args)
        {
            var gys = args.Through();
            var (x0, x1) = (Inputs.ElementAt(0), Inputs.ElementAt(1));
            var gx0 = new Variable(gys.Single().Data / x1.Data);
            var gx1 = new Variable(gys.Single().Data * (-x0.Data / x1.Data.pow(2)));
            if (x0.Shape != x1.Shape)
            {
                gx0 = SumTo.Invoke(gx0, x0.Shape).Single();
                gx1 = SumTo.Invoke(gx1, x1.Shape).Single();
            }

            return [gx0, gx1];
        }

        public static Variable[] Invoke(Variable x0, Variable x1)
        {
            return new Div().Call(Params<Variable, Variable>.args(x0, x1));
        }

        public static Variable[] ReverseInvoke(Variable x0, Variable x1)
        {
            return new Div().Call(OrderedParams<Variable, Variable>.args(x1, x0));
        }
    }
}
