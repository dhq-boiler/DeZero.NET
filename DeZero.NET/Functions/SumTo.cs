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

        public override Variable[] Forward(params Variable[] xs)
        {
            X_Shape = xs[0].Shape;
            var y = Utils.sum_to(xs[0].Data, Shape);
            return [new Variable(y)];
        }

        public override Variable[] Backward(params Variable[] gys)
        {
            var gx = BroadcastTo.Invoke(gys.Single(), X_Shape);
            return gx;
        }

        public static Variable[] Invoke(Variable x, Shape shape)
        {
            if (x.Shape == shape)
            {
                return [Utils.as_variable(x)];
            }
            return new SumTo(shape).Forward(x);
        }
    }
}
