using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class StdDev : Function
    {
        private readonly int[] _axis;
        private readonly bool _keepdims;
        private const float EPSILON = 1e-7f;

        public StdDev(int[] axis = null, bool keepdims = false)
        {
            _axis = axis;
            _keepdims = keepdims;
        }

        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>(0);

            using var axis = new Axis(_axis);
            using var mean = x.Data.Value.mean(axis, keepdims: true);

            // 平均からの偏差の二乗を計算
            using var diff = x.Data.Value - mean;
            using var squared_diff = diff * diff;

            // 分散を計算
            using var variance = squared_diff.mean(axis, keepdims: _keepdims);

            // 標準偏差を計算（分散の平方根）
            using var std = xp.sqrt(variance + EPSILON);

            return [std.copy().Relay(this)];
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Get<Variable>(0);
            var x = Inputs.ElementAt(0).Variable;

            using var axis = new Axis(_axis);
            using var mean = x.Data.Value.mean(axis, keepdims: true);
            using var diff = x.Data.Value - mean;

            // 標準偏差の計算（Forward処理と同じ）
            using var squared_diff = diff * diff;
            using var variance = squared_diff.mean(axis, keepdims: _keepdims);
            using var a = variance + EPSILON;
            using var std = xp.sqrt(a);

            // 要素数を取得
            int n = 1;
            if (_axis != null)
            {
                foreach (var dim in x.Shape.Dimensions)
                {
                    n *= dim;
                }
                foreach (var ax in _axis)
                {
                    n /= x.Shape.Dimensions[ax];
                }
            }
            else
            {
                n = x.Data.Value.size;
            }

            // 勾配を計算
            // δstd/δx = (x - μ)/(n * std)
            using var b = std + EPSILON;
            using var c = n * b;
            using var gx = diff / c;

            // ブロードキャストの処理
            if (_axis != null && !_keepdims)
            {
                gy = Functions.BroadcastTo.Invoke(gy, gx.shape.Dimensions)[0];
            }

            return [gy * gx];
        }

        public static Variable[] Invoke(Variable x, int[] axis = null, bool keepdims = false)
        {
            return new StdDev(axis, keepdims).Call(Params.New.SetPositionalArgs(x));
        }
    }
}
