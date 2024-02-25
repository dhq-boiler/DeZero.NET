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

        public override Variable[] Forward(params Variable[] xs)
        {
            X_Shape = xs[0].Shape;
            var y = xp.broadcast_to(xs.Select(x => x.Data).Single(), Shape);
            return [new Variable(y)];
        }

        public override Variable[] Backward(params Variable[] gys)
        {
            var gx = SumTo.Invoke(gys.Single(), Shape);
            return gx;
        }

        public static Variable[] Invoke(Variable x, Shape shape)
        {
            if (x.Shape == shape)
            {
                return [Utils.as_variable(x)];
            }
            return new BroadcastTo(shape).BaseForward(x);
        }
    }
}
