using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class Affine : Function
    {
        public Variable W { get; set; }
        public Variable b { get; set; }

        public Affine(int input_size, int output_size)
        {
            W = xp.random.randn(input_size, output_size).ToVariable();
            b = xp.zeros(output_size).ToVariable();
        }

        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>(0);
            var y = xp.dot(x.Data.Value, W.Data.Value) + b.Data.Value;
            return [y.ToVariable(this)];
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Get<Variable>(0);
            W.Grad.Value = xp.dot(Inputs.ElementAt(0).NDarray.T, gy.Data.Value).ToVariable();
            b.Grad.Value = gy.Data.Value.sum(axis: 0).ToVariable();
            var gx = xp.dot(gy.Data.Value, W.Data.Value.T).ToVariable();
            return [gx];
        }

        public static Variable Invoke(Variable x, int input_size, int output_size)
        {
            return new Affine(input_size, output_size).Call(Params.New.SetPositionalArgs(x))[0];
        }
    }
}
