using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DeZero.NET.Core;

namespace DeZero.NET.Functions
{
    public class SoftmaxCrossEntropy : Function
    {
        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>("x");
            var t = args.Get<Variable>("t");
            var N = x.Shape[0];
            var log_z = Utils.logsumexp(x, axis: [1]);
            var log_p = x - log_z;
            log_p = log_p.Data[xp.arange(N), t.Data.ravel()].ToVariable(this);
            var y = (-log_p.Data).sum() / (float)N;
            return [y.ToVariable(this)];
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
            var t_onehot = xp.eye(CLS_NUM, dtype: t.Dtype)[t.Data].ToVariable();
            y = (y - t_onehot) * gy;
            return [y];
        }

        public static Variable[] Invoke(Variable x, Variable t)
        {
            return new SoftmaxCrossEntropy().Call(Params.New.SetPositionalArgs(x, t));
        }
    }
}
