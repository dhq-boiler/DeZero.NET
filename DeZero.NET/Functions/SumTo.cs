﻿namespace DeZero.NET.Functions
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
            var y = xs.Select(x => new Variable(Utils.sum_to(x.Data, Shape))).ToArray();
            return y;
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
            return new SumTo(shape).BaseForward(x);
        }
    }
}
