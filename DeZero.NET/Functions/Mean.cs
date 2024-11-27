using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class Mean : Function
    {
        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>(0);
            var axis = args.Get<int?>("axis");
            var keepdims = args.Get<bool?>("keepdims");
            var y = x.Data.Value.mean(axis: axis, keepdims: keepdims);
            return [y.ToVariable(this)];
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Get<Variable>(0);
            var x = Inputs.ElementAt(0).Variable;
            var axis = Inputs.ElementAt(1).Value as int?;
            var keepdims = Inputs.ElementAt(2).Value as bool?;
            var gx = gy.Data.Value;

            // axis が指定されている場合は、その軸に沿った要素数で割る
            if (axis.HasValue)
            {
                using var x_shape = x.Shape;
                // 入力の shape から指定された軸の要素数を取得
                int axisSize = x_shape[axis.Value];

                // keepdims が false の場合、gyの形状を復元する必要がある
                if (keepdims != true)
                {
                    using var gy_shape = gy.Shape;
                    // 削除された次元を復元
                    var _gx = gx.reshape(GetExpandedShape(gy_shape.Dimensions, x_shape.Dimensions, axis.Value));
                    gx.Dispose();
                    gx = _gx;
                }

                // 指定された軸の要素数で割る
                var __gx = gx / axisSize;
                gx.Dispose();
                gx = __gx;
            }
            else
            {
                // axis が指定されていない場合は全要素の平均
                var n = x.Data.Value.size;
                var _gx = gx / n;
                gx.Dispose();
                gx = _gx;
            }

            // broadcast_to で入力と同じ形状に拡張
            var ___gx = xp.broadcast_to(gx, x.Data.Value.shape);
            gx.Dispose();
            gx = ___gx;

            return [gx.ToVariable(this)];
        }

        // 削除された次元を復元するためのヘルパーメソッド
        private int[] GetExpandedShape(int[] gyShape, int[] xShape, int axis)
        {
            var result = new List<int>();
            int gyIndex = 0;

            // 元の形状の各次元に対して処理
            for (int i = 0; i < xShape.Length; i++)
            {
                if (i == axis)
                {
                    // 平均を取った軸の次元を復元
                    result.Add(xShape[i]);
                }
                else
                {
                    // その他の次元はgyShapeから取得
                    result.Add(gyShape[gyIndex]);
                    gyIndex++;
                }
            }

            return result.ToArray();
        }

        public static Variable[] Invoke(Variable x, int? axis = null, bool? keepdims = null)
        {
            return new Mean().Call(Params.New.SetPositionalArgs(x).SetKeywordArg(axis, keepdims));
        }
    }
}
