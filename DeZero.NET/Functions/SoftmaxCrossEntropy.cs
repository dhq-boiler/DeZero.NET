using DeZero.NET.Core;

namespace DeZero.NET.Functions
{
    public class SoftmaxCrossEntropy : Function
    {
        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>(0);
            var t = args.Get<Variable>(1);
            var N = x.Shape[0];
            using var log_z = Utils.logsumexp(x, axis: [1]);
            using var log_p = x - log_z;
            using var log_p2 = GetItem.Invoke(log_p, xp.arange(N), t.Data.Value.ravel())[0];
            var y = Div.Invoke(Neg.Invoke(Sum.Invoke(log_p2)[0])[0], new NDarray((float)N).ToVariable(this))[0]; 
            return [y];
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Get<Variable>(0);
            var x = Inputs.ElementAt(0).Variable;
            var t = Inputs.ElementAt(1).Variable;
            var N = x.Shape[0];
            var CLS_NUM = x.Shape[1];

            gy *= 1f / N;
            var y = Softmax.Invoke(x)[0];
            using var t_onehot = xp.eye(CLS_NUM, dtype: t.Dtype)[t.Data.Value].ToVariable();
            y = (y - t_onehot) * gy;
            return [y];
        }

        public static Variable[] Invoke(Variable x, Variable t)
        {
            return new SoftmaxCrossEntropy().Call(Params.New.SetPositionalArgs(x, t));
        }
    }
}
