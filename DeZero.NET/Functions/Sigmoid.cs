using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class Sigmoid : Function
    {
        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>("x");
            try
            {
                // 入力値をコピーして0.5を掛け、tanhを適用
                using var halfX = x.Data.Value.copy() * 0.5;
                using var tanhResult = xp.tanh(halfX);
                // 結果を0.5倍して0.5を足す
                using var __y = tanhResult * 0.5;
                using var _y = __y + 0.5;
                using var y = _y.ToVariable(this);
                return [y.copy().Relay(this)];
            }
            catch (Exception ex)
            {
                throw new Exception($"Error in Sigmoid Forward: {ex.Message}", ex);
            }
        }

        public override Variable[] Backward(Params args)
        {
            try
            {
                var gy = args.Get<Variable>(0);
                var y = Outputs.ElementAt(0);

                // 1をNDarrayとして作成
                using var one = new NDarray(1).ToVariable();

                // (-y + 1) の計算
                using var negY = -y;
                using var yPlusOne = negY + one;

                // y * (1 - y) の計算
                using var yTimesOneMunusY = y * yPlusOne;

                // 最終的な勾配の計算
                using var gx = gy * yTimesOneMunusY;

                return [gx.copy()];
            }
            catch (Exception ex)
            {
                throw new Exception($"Error in Sigmoid Backward: {ex.Message}", ex);
            }
        }

        public static Variable[] Invoke(Variable x)
        {
            return new Sigmoid().Call(Params.New.SetPositionalArgs(x));
        }
    }
}
