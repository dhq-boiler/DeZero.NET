namespace DeZero.NET.Functions
{
    public class Div : Function
    {
        public static Func<Variable[], Variable[]> F => x => [(x[0].Data / x[1].Data).ToVariable()];

        public Div()
        { }

        public Div(Func<Variable[], Variable[]> f)
            : base(f)
        { }

        public override Variable[] Forward(params Variable[] xs)
        {
            var y = F([xs[0], xs[1]])[0];
            return [y];
        }

        public override Variable[] Backward(params Variable[] gys)
        {
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
            return new Div().BaseForward(x0, x1);
        }

        public static Variable[] ReverseInvoke(Variable x0, Variable x1)
        {
            return new Div().BaseForward(x1, x0);
        }
    }
}
