﻿using DeZero.NET.Core;
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
            var inter = xp.concatenate(y.ToArray());
            return [inter.Relay(this)];
        }

        public override Variable[] Backward(Params args)
        {
            var xs = Inputs;
            var c = C;
            var gx = xs.Select(x => c * x.Variable.Data.Value.pow(c - 1) * args.Through.Single().Variable.Data.Value);
            var inter = xp.concatenate(gx.ToArray());
            return [inter.ToVariable(this)];
        }

        public static Variable[] Invoke(Variable x, Variable c)
        {
            return new Pow(c.Data.Value.asscalar<double>()).Call(Params.New.SetPositionalArgs(x));
        }
    }
}
