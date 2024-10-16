using DeZero.NET.Core;
using DeZero.NET.Extensions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeZero.NET.Functions
{
    public class Stack : Function
    {
        private int _axis;

        public override Variable[] Forward(Params args)
        {
            var inputs = args.Get<Variable[]>("inputs");
            _axis = args.Get<int>("axis");

            // すべての入力の形状が同じであることを確認
            var shape = inputs[0].Shape;
            if (inputs.Any(x => !x.Shape.Dimensions.SequenceEqual(shape.Dimensions)))
            {
                throw new ArgumentException("All inputs must have the same shape");
            }

            // 新しい形状を計算
            var newShape = shape.Dimensions.ToArray();
            if (_axis < 0) _axis += newShape.Length + 1;
            if (_axis < 0 || _axis > newShape.Length)
            {
                throw new ArgumentException("Invalid axis");
            }
            newShape = newShape.Take(_axis).Concat(new[] { inputs.Length }).Concat(newShape.Skip(_axis)).ToArray();

            // 新しいNDArrayを作成
            var y = xp.empty(newShape, inputs[0].Dtype);

            // データをコピー
            for (int i = 0; i < inputs.Length; i++)
            {
                var slices = Enumerable.Range(0, newShape.Length)
                    .Select(j => j == _axis ? new Slice(i, i + 1) : new Slice(null, null))
                    .ToArray();
                y.SetSlice(slices, inputs[i].Data.Value);
            }

            return new[] { y.ToVariable(this) };
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Through[0];
            var inputs = Inputs.Select(x => x.Variable).ToArray();

            var gxs = new Variable[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
            {
                var slices = Enumerable.Range(0, gy.Variable.ndim)
                    .Select(j => j == _axis ? new Slice(i, i + 1) : new Slice(null, null))
                    .ToArray();
                gxs[i] = SliceFunc.Invoke(gy.Variable, slices)[0];
                if (gxs[i].ndim > inputs[i].ndim)
                {
                    gxs[i] = Squeeze.Invoke(gxs[i], _axis)[0];
                }
            }

            return gxs;
        }

        public static Variable[] Invoke(Variable[] inputs, int axis = 0)
        {
            return new Stack().Call(Params.New.SetKeywordArg("inputs", inputs).SetKeywordArg("axis", axis));
        }
    }
}
