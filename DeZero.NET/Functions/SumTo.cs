using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class SumTo : Function
    {
        public Shape Shape { get; }
        public Shape X_Shape { get; set; }


        public SumTo(Shape shape)
        {
            Shape = shape;
        }

        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>(0);
            X_Shape = x.Shape;
            var y = Utils.sum_to(x.Data.Value, Shape);
            return [y.Relay(this)];
        }

        public override Variable[] Backward(Params args)
        {
            using var gx = BroadcastTo.Invoke(args.Through.Single().Variable, X_Shape)[0];
            return [gx.copy()];
        }

        public static Variable[] Invoke(Variable x, Shape shape)
        {
            if (x.Shape == shape)
            {
                return [x.copy()];
            }
            return new SumTo(shape).Call(Params.New.SetPositionalArgs(x));
        }
    }
}
