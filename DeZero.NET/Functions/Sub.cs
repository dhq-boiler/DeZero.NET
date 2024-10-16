using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class Sub : Function
    {
        public Shape X0_Shape { get; set; }
        public Shape X1_Shape { get; set; }

        public override Variable[] Forward(Params args)
        {
            var x0 = args.Get<Variable>(0);
            var x1 = args.Get<Variable>(1);
            X0_Shape = x0.Shape;
            X1_Shape = x1.Shape;
            var y = x0.Data.Value - x1.Data.Value;
            return [new Variable(y)];
        }

        public override Variable[] Backward(Params args)
        {
            var gys = args.Through;
            var gy = gys[0].Variable;
            var gx0 = gy;
            var gx1 = (-gy.Data.Value).ToVariable();
            if (X0_Shape != X1_Shape)
            {
                gx0 = SumTo.Invoke(gx0, X0_Shape).Single();
                gx1 = SumTo.Invoke(gx1, X1_Shape).Single();
            }

            return [gx0, gx1];
        }

        public static Variable[] Invoke(Variable x0, Variable x1)
        {
            return new Sub().Call(Params.New.SetPositionalArgs(x0, x1));
        }
        public static Variable[] ReverseInvoke(Variable x0, Variable x1)
        {
            return new Sub().Call(OrderedParams<Variable, Variable>.args(x1, x0));
        }
    }
}
