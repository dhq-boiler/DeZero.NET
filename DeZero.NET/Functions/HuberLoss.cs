using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class HuberLoss : Function, IDisposable
    {
        private readonly float _delta;
        private NDarray _mask; // quadratic lossを適用する領域のマスク

        public HuberLoss(float delta = 1.0f)
        {
            _delta = delta;
        }

        public override Variable[] Forward(Params args)
        {
            var x0 = args.Get<Variable>(0); // predictions
            var x1 = args.Get<Variable>(1); // targets

            using var diff = x0.Data.Value - x1.Data.Value;
            using var abs_diff = xp.abs(diff);

            // |diff| <= deltaの領域のマスクを作成
            _mask = abs_diff <= _delta;

            using var diff2 = diff * diff;
            using var squared_loss = 0.5f * diff2;
            using var a = _delta * abs_diff;
            using var linear_loss = a - 0.5f * _delta * _delta;

            // マスクを使って二つの損失を組み合わせる
            using var combined_loss = xp.where(_mask, squared_loss, linear_loss);
            using var mean_loss = combined_loss.mean();

            return [mean_loss.copy().Relay(this)];
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Get<Variable>(0);
            var x0 = Inputs.ElementAt(0).Variable;
            var x1 = Inputs.ElementAt(1).Variable;

            using var diff = x0.Data.Value - x1.Data.Value;
            using var abs_diff = xp.abs(diff);

            // 勾配の計算
            using var grad_quadratic = diff;  // squared lossの勾配
            using var signed = xp.sign(diff);
            using var grad_linear = _delta * signed;  // linear lossの勾配

            // マスクを使って勾配を組み合わせる
            using var combined_grad = xp.where(_mask, grad_quadratic, grad_linear);

            // バッチサイズで正規化
            var batch_size = x0.Shape[0];
            using var a = (gy * combined_grad);
            using var gx0 = (a / batch_size);
            using var gx1 = (-gx0);

            return [gx0.copy(), gx1.copy()];
        }

        public static Variable[] Invoke(Variable x0, Variable x1, float delta = 1.0f)
        {
            return new HuberLoss(delta).Call(Params.New.SetPositionalArgs(x0, x1));
        }

        protected void Dispose(bool disposing)
        {
            if (disposing)
            {
                _mask?.Dispose();
            }
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }
    }
}
