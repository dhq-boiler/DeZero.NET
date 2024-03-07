using DeZero.NET.Core;

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

        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>("x");
            X_Shape = x.Shape;
            var y = x.Data.reshape(Shape);
            return [new Variable(y)];
        }

        public override Variable[] Backward(Params args)
        {
            return Invoke(args.Through()[0], X_Shape);
        }
        
        public static Variable[] Invoke(Variable x, Shape shape)
        {
            if (x.Shape == shape)
            {
                return [x];
            }
            return new Reshape(shape).BaseForward(Params<Variable>.args(x));
        }
    }
}
