namespace DeZero.NET.Functions
{
    public class Reshape : Function
    {
        public Shape Shape { get; set; }
        public Shape X_Shape { get; set; }

        public Reshape(Shape shape)
        {
            Shape = shape;
        }

        public override Variable[] Forward(params Variable[] xs)
        {
            X_Shape = xs[0].Shape;
            var inter = xp.concatenate(xs.Select(x => x.Data).ToArray());
            var y = inter.reshape(Shape);
            return [new Variable(y)];
        }

        public override Variable[] Backward(params Variable[] gys)
        {
            return Invoke(gys[0], X_Shape);
        }
        
        public static Variable[] Invoke(Variable x, Shape shape)
        {
            if (x.Shape == shape)
            {
                return [x];
            }
            return new Reshape(shape).BaseForward(x);
        }
    }
}
