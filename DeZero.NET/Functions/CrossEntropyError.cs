using DeZero.NET.Extensions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DeZero.NET.Core;

namespace DeZero.NET.Functions
{
    public class CrossEntropyError : Function
    {
        private const float EPSILON = 1e-7f;

        public override Variable[] Forward(Params args)
        {
            var x0 = args.Get<Variable>(0);  // 予測値
            var x1 = args.Get<Variable>(1);  // 正解ラベル

            using var eps = new NDarray(EPSILON);
            using var oneMinusEps = new NDarray(1.0f - EPSILON);

            // クリッピングして数値安定性を確保
            using var clipped_x0 = x0.Data.Value.clip(eps, oneMinusEps);

            // クロスエントロピーの計算: -Σ(t * log(y))
            using var log_x0 = clipped_x0.log();
            using var a = x1.Data.Value * log_x0;
            using var b = -a;
            using var c = b.sum();
            using var y = c / x0.Shape[0];
            
            return [y.copy().Relay(this)];
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Get<Variable>(0);
            var x0 = Inputs.ElementAt(0).Variable;  // 予測値
            var x1 = Inputs.ElementAt(1).Variable;  // 正解ラベル

            var batch_size = x0.Shape[0];

            using var eps = new NDarray(EPSILON);
            using var oneMinusEps = new NDarray(1.0f - EPSILON);

            // クリッピングして数値安定性を確保
            using var clipped_x0 = x0.Data.Value.clip(eps, oneMinusEps);

            using var a = -x1.Data.Value;
            using var b = a / clipped_x0;
            using var c = gy * b;

            // 勾配の計算: -t/y
            using var gx0 = c / batch_size;

            // x1（正解ラベル）に関する勾配は使用しない
            using var gx1 = xp.zeros_like(x1.Data.Value).ToVariable();

            return [gx0.copy(), gx1.copy()];
        }

        public static Variable[] Invoke(Variable x0, Variable x1)
        {
            return new CrossEntropyError().Call(Params.New.SetPositionalArgs(x0, x1));
        }
    }
}
