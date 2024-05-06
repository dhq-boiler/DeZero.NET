using DeZero.NET.Core;

namespace DeZero.NET.Functions
{
    public class Linear : Function
    {
        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>("x");
            var W = args.Get<Variable>("W");
            var b = args.Get<Variable>("b");
            var y = x.Data.dot(W.Data);
            if (b?.Data is not null)
            {
                y += b.Data;
            }

            return [y.ToVariable(this)];
        }

        public override Variable[] Backward(Params args)
        {
            var gys = args.Through();
            var gy = gys[0];
            var x = Inputs.ElementAt(0);
            var W = Inputs.ElementAt(1);
            var b = Inputs.ElementAt(2);
            var gb = b.Variable.Data is null ? null : SumTo.Invoke(gy.Variable, b.Variable.Shape)[0];
            var gx = MatMul.Invoke(gy.Variable, W.Variable.T)[0];
            var gW = MatMul.Invoke(x.Variable.T, gy.Variable)[0];
            return [gx, gW, gb];
        }

        public static Variable[] Invoke(Variable x, Variable W, Variable b = null)
        {
            return new Linear().Call(Params<Variable, Variable, Variable>.args(x, W, b));
        }
    }
}
