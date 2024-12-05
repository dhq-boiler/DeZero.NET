using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class ReLU : Function
    {
        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>("x");
            var y = xp.maximum(x.Data.Value, new NDarray(0.0f)).ToVariable(this);
            return [y.Relay(this)];
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Get<Variable>(0);
            var x = Inputs.ElementAt(0).Variable;

            // マスクを計算 (x > 0)
            using var mask = x.Data.Value > 0;

            // gx を x と同じサイズで初期化（0で埋める）
            var gx_array = xp.zeros_like(x.Data.Value);

            // gy の値を gx の対応する位置にコピー
            // これにより、gx は x と同じシェイプを維持しつつ、
            // gy の値が正しい位置に配置される
            gx_array[new Slice(), new Slice(null, gy.Shape[1])] = gy.Data.Value;

            // マスクを適用
            var gx = (gx_array * mask).ToVariable(this);

            return [gx];
        }

        public static Variable[] Invoke(Variable x)
        {
            return new ReLU().Call(Params.New.SetPositionalArgs(x));
        }
    }
}