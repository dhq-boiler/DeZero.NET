using DeZero.NET.Core;

namespace DeZero.NET.Functions
{
    public class BroadcastTo : Function
    {
        public Shape Shape { get; }
        public Shape X_Shape { get; set; }


        public BroadcastTo(Shape shape)
        {
            Shape = shape;
        }

        public override Variable[] Forward(Params args)
        {
            var xs = args.Through();
            X_Shape = xs[0].Shape;
            var y = xp.broadcast_to(xs.Select(x => x.Data).Single(), Shape);
            return [new Variable(y)];
        }

        public override Variable[] Backward(Params args)
        {
            var gys = args.Through();
            var gx = SumTo.Invoke(gys.Single(), Shape);
            return gx;
        }

        public static Variable[] Invoke(Variable x, Shape shape)
        {
            if (x.Shape == shape)
            {
                return [Utils.as_variable(x)];
            }
            return new BroadcastTo(shape).BaseForward(Params<Variable>.args(x));
        }
    }
}
