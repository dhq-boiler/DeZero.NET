using DeZero.NET.Core;

namespace DeZero.NET.Functions
{
    public class LogSoftmax : Function
    {
        public int[] Axis { get; }

        public LogSoftmax(int[] axis = null)
        {
            if (axis is null)
            {
                Axis = [1];
            }
            else
            {
                Axis = axis;
            }
        }

        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>(0);
            var log_z = Utils.logsumexp(x, axis: Axis);
            var y = x - log_z;
            return [y];
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Get<Variable>(0);
            var y = Outputs.ElementAt(0);
            var gx = gy - xp.exp(y.Data.Value) * gy.Data.Value.sum(axis: new Axis(Axis), keepdims: true);
            return [gx];
        }

        public static Variable[] Invoke(Variable x, int[] axis = null)
        {
            if (axis is null)
            {
                axis = [1];
            }
            return new LogSoftmax(axis).Call(Params.New.SetPositionalArgs(x));
        }
    }
}
