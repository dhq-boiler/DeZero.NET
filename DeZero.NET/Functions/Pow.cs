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
            var y = xs.Select(x =>x.Data.pow(C));
            var inter = xp.concatenate(y.ToArray());
            return [new Variable(inter)];
        }

        public override Variable[] Backward(params Variable[] gys)
        {
            var xs = Inputs;
            var c = C;
            var gx = xs.Select(x => c * x.Data.pow(c - 1) * gys.Single().Data);
            var inter = xp.concatenate(gx.ToArray());
            return [new Variable(inter)];
        }

        public static Variable[] Invoke(Variable x, Variable c)
        {
            return new Pow(c.Data.asscalar<double>()).BaseForward(x);
        }
    }
}
