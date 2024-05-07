using DeZero.NET.Core;

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
            var y = Utils.sum_to(x.Data, Shape);
            return [y.ToVariable()];
        }

        public override Variable[] Backward(Params args)
        {
            var gx = BroadcastTo.Invoke(args.Through().Single().Variable, X_Shape);
            return gx;
        }

        public static Variable[] Invoke(Variable x, Shape shape)
        {
            if (x.Shape == shape)
            {
                return [Utils.as_variable(x)];
            }
            return new SumTo(shape).Call(Params.New.SetPositionalArgs(x));
        }
    }
}
