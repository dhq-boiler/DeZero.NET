using DeZero.NET.Core;

namespace DeZero.NET.Functions
{
    public class MatMul : Function
    {
        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>("x");
            var W = args.Get<Variable>("W");
            var y = x.Data.dot(W.Data);
            return [y.ToVariable()];
        }

        public override Variable[] Backward(Params args)
        {
            var gys = args.Through();
            var gy = gys[0].Variable;
            var x = Inputs.ElementAt(0).Variable;
            var W = Inputs.ElementAt(1).Variable;
            var gx = MatMul.Invoke(gy, W.T)[0];
            var gW = MatMul.Invoke(x.T, gy)[0];
            return [gx, gW];
        }

        public static Variable[] Invoke(Variable x, Variable W)
        {
            return new MatMul().Call(Params.New.SetKeywordArg(x, W));
        }
    }
}
