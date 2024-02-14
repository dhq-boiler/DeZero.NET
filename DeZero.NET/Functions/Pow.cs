namespace DeZero.NET.Functions
{
    public class Pow : Function
    {
        public double C { get; }

        public Pow(double c)
        {
            C = c;
        }

        public override Variable[] Forward(params Variable[] xs)
        {
            var y = xs[0].Data.pow(C);
            return [new Variable(y)];
        }

        public override Variable[] Backward(params Variable[] gys)
        {
            var x = Inputs.ElementAt(0);
            var c = C;
            var gx = c * x.Data.pow(c - 1) * gys.Single().Data;
            return [new Variable(gx)];
        }

        public static Variable[] Invoke(Variable x, Variable c)
        {
            return new Pow(c.Data.asscalar<double>()).Forward(x);
        }
    }
}
