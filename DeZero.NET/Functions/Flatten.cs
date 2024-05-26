using DeZero.NET.Core;

namespace DeZero.NET.Functions
{
    public class Flatten : Function
    {
        public Shape OriginalShape { get; set; }

        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>(0);
            this.OriginalShape = x.Shape;
            return Reshape.Invoke(x, new Shape(x.Shape[0], -1));
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Get<Variable>(0);
            return Reshape.Invoke(gy, this.OriginalShape);
        }

        public static Variable Invoke(Variable x)
        {
            return new Flatten().Call(Params.New.SetPositionalArgs(x))[0];
        }
    }
}
