using DeZero.NET.Core;
using DeZero.NET.Extensions;

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
            var y = args.Through.Select(x => x.Variable.Data.Value.pow(C));
            using var inter = xp.concatenate(y.ToArray());
            return [inter.copy().Relay(this)];
        }

        public override Variable[] Backward(Params args)
        {
            var xs = Inputs;
            var c = C;
            var gx = xs.Select(x => c * x.Variable.Data.Value.pow(c - 1)).ToList();
            var _gx = gx.Select(x => x * args.Through.Single().Variable.Data.Value).ToList();
            gx.ToList().ForEach(x => x.Dispose());
            using var inter = xp.concatenate(_gx.ToArray());
            _gx.ToList().ForEach(x => x.Dispose());
            return [inter.copy().ToVariable(this)];
        }

        public static Variable[] Invoke(Variable x, Variable c)
        {
            return new Pow(c.Data.Value.asscalar<double>()).Call(Params.New.SetPositionalArgs(x));
        }
    }
}
