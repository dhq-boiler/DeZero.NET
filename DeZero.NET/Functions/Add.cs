using DeZero.NET.Core;

namespace DeZero.NET.Functions
{
    public class Add : Function
    {
        public static Func<Params, Variable[]> F => x => [(x.Get<Variable>("x0").Data.Value + x.Get<Variable>("x1").Data.Value).ToVariable()];
        public Shape X0_Shape { get; set; }
        public Shape X1_Shape { get; set; }

        public Add()
        { }

        public Add(Func<Params, Variable[]> f)
            : base(f)
        { }

        public override Variable[] Forward(Params args)
        {
            var xs = args.Through;
            X0_Shape = xs[0].Variable.Shape;
            X1_Shape = xs[1].Variable.Shape;
            var x0 = xs[0].Value;
            var x1 = xs[1].Value;
            var y = F(Params.New.SetPositionalArgs(x0, x1))[0];
            return [y];
        }

        public override Variable[] Backward(Params args)
        {
            var gys = args.Through;
            var gx0 = gys[0].Variable;
            var gx1 = gys[0].Variable;
            if (X0_Shape != X1_Shape)
            {
                gx0 = SumTo.Invoke(gx0, X0_Shape).Single();
                gx1 = SumTo.Invoke(gx1, X1_Shape).Single();
            }

            return [gx0, gx1];
        }

        public static (Variable[], Add) Invoke(Variable x0, Variable x1)
        {
            var op = new Add();
            return (op.Call(Params.New.SetPositionalArgs(x0, x1)), op);
        }

        public static Variable[] Invoke(Add op, Variable x0, Variable x1)
        {
            return op.Call(Params.New.SetPositionalArgs(x0, x1));
        }
    }
}
