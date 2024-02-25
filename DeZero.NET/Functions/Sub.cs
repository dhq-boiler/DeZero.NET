namespace DeZero.NET.Functions
{
    public class Sub : Function
    {
        public Shape X0_Shape { get; set; }
        public Shape X1_Shape { get; set; }

        public override Variable[] Forward(params Variable[] xs)
        {
            X0_Shape = xs[0].Shape;
            X1_Shape = xs[1].Shape;
            var y = xs[0].Data - xs[1].Data;
            return [new Variable(y)];
        }

        public override Variable[] Backward(params Variable[] gys)
        {
            var gx0 = gys[0];
            var gx1 = -gys[0];
            if (X0_Shape != X1_Shape)
            {
                gx0 = SumTo.Invoke(gx0, X0_Shape).Single();
                gx1 = SumTo.Invoke(gx1, X1_Shape).Single();
            }

            return [gx0, gx1];
        }

        public static Variable[] Invoke(Variable x0, Variable x1)
        {
            return new Sub().BaseForward(x0, x1);
        }
        public static Variable[] ReverseInvoke(Variable x0, Variable x1)
        {
            return new Sub().BaseForward(x1, x0);
        }
    }
}
