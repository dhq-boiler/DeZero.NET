namespace DeZero.NET.Functions
{
    public class Mul : Function
    {
        public Shape X0_Shape { get; set; }
        public Shape X1_Shape { get; set; }

        public override Variable[] Forward(params Variable[] xs)
        {
            X0_Shape = xs[0].Shape;
            X1_Shape = xs[1].Shape;
            var y = xs[0].Data * xs[1].Data;
            return [new Variable(y)];
        }

        public override Variable[] Backward(params Variable[] gys)
        {
            var x0 = this.Inputs.ElementAt(0);
            var x1 = this.Inputs.ElementAt(1);
            var gx0 = new Variable(gys.Single().Data * x1.Data);
            var gx1 = new Variable(gys.Single().Data * x0.Data);
            if (X0_Shape != X1_Shape)
            {
                gx0 = SumTo.Invoke(gx0, X0_Shape).Single();
                gx1 = SumTo.Invoke(gx1, X1_Shape).Single();
            }

            return [gx0, gx1];
        }

        public static Variable[] Invoke(Variable x0, Variable x1)
        {
            return new Mul().Forward(x0, x1);
        }
    }
}
