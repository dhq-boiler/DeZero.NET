﻿using DeZero.NET.Core;

namespace DeZero.NET.Functions
{
    public class Mul : Function
    {
        public static Func<Params, Variable[]> F => x => [(x.Get<Variable>("x0").Data * x.Get<Variable>("x1").Data).ToVariable()];
        public Shape X0_Shape { get; set; }
        public Shape X1_Shape { get; set; }

        public Mul()
        { }

        public Mul(Func<Params, Variable[]> f)
            : base(f)
        { }

        public override Variable[] Forward(Params args)
        {
            var xs = args.Through();
            var x0 = xs[0];
            var x1 = xs[1];
            X0_Shape = x0.Shape;
            X1_Shape = x1.Shape;
            var y = F(Params<Variable, Variable>.args(x0, x1))[0];
            return [y];
        }

        public override Variable[] Backward(Params args)
        {
            var gys = args.Through();
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
            return new Mul().BaseForward(Params<Variable, Variable>.args(x0, x1));
        }
    }
}
