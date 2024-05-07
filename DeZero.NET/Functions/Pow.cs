using DeZero.NET.Core;

namespace DeZero.NET.Functions
{
    public class Pow : Function
    {
        public double C { get; }

        public Pow(double c)
        {
            C = c;
        }

        public override Variable[] Forward(Params args)
        {
            var y = args.Through().Select(x => x.Variable.Data.pow(C));
            var inter = xp.concatenate(y.ToArray());
            return [new Variable(inter)];
        }

        public override Variable[] Backward(Params args)
        {
            var xs = Inputs;
            var c = C;
            var gx = xs.Select(x => c * x.Variable.Data.pow(c - 1) * args.Through().Single().Variable.Data);
            var inter = xp.concatenate(gx.ToArray());
            return [new Variable(inter)];
        }

        public static Variable[] Invoke(Variable x, Variable c)
        {
            return new Pow(c.Data.asscalar<double>()).Call(Params.New.SetPositionalArgs(x));
        }
    }
}
