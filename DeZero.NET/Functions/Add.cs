namespace DeZero.NET.Functions
{
    public class Add : Function
    {
        public static Func<Variable[], Variable[]> F => x => [(x[0].Data + x[1].Data).ToVariable()];
        public Shape X0_Shape { get; set; }
        public Shape X1_Shape { get; set; }

        public Add()
        { }

        public Add(Func<Variable[], Variable[]> f)
            : base(f)
        { }

        public override Variable[] Forward(params Variable[] xs)
        {
            X0_Shape = xs[0].Shape;
            X1_Shape = xs[1].Shape;
            var y = F([xs[0], xs[1]])[0];
            return [y];
        }

        public override Variable[] Backward(params Variable[] gys)
        {
            var gx0 = gys[0];
            var gx1 = gys[0];
            if (X0_Shape != X1_Shape)
            {
                gx0 = SumTo.Invoke(gx0, X0_Shape).Single();
                gx1 = SumTo.Invoke(gx1, X1_Shape).Single();
            }

            return [gx0, gx1];
        }

        public static Variable[] Invoke(Variable x0, Variable x1)
        {
            return new Add().BaseForward(x0, x1);
        }
    }
}
