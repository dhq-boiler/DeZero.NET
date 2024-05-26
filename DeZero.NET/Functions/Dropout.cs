using DeZero.NET.Core;
using Numpy;

namespace DeZero.NET.Functions
{
    public class Dropout : Function
    {
        public double DropoutRatio { get; set; }
        public NDarray Mask { get; set; }

        public Dropout(double dropoutRatio)
        {
            this.DropoutRatio = dropoutRatio;
        }

        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>(0);
            if (Config.Train)
            {
                var z = new NDarray(np.random.rand(x.Shape.Dimensions));
                Mask = z > DropoutRatio;
                var scale = xp.array(1.0 - DropoutRatio).astype(x.Dtype);
                var y = x * Mask / scale;
                return [y];
            }
            else
            {
                return [x];
            }
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Get<Variable>(0);
            return [gy * Mask];
        }

        public static Variable Invoke(Variable x, double dropoutRatio = 0.5)
        {
            return new Dropout(dropoutRatio).Call(Params.New.SetPositionalArgs(x))[0];
        }
    }
}
