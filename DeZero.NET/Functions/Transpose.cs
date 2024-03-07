﻿using DeZero.NET.Core;

namespace DeZero.NET.Functions
{
    public class Transpose : Function
    {
        public Axis[] Axes { get; set; }

        public Transpose(params Axis[] axes)
        {
            Axes = axes;
        }

        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>("x").Data; // 仮定: 入力は単一のVariableオブジェクト
            var y = x.transpose(Axes.SelectMany(ax => ax.Axes).ToArray()); // xp.Transposeを使用して行列を転置
            return [new Variable(y)];
        }

        public override Variable[] Backward(Params args)
        {
            var gys = args.Through();
            if (Axes is null)
            {
                return Invoke(gys[0]);
            }

            var axes_len = Axes.Length;
            var inv_axes = Enumerable.Range(0, axes_len).Select(i => Axes.ToList().IndexOf(new Axis([i]))).ToArray();
            return gys.Select(gy => new Variable(xp.transpose(gy.Data, inv_axes))).ToArray();
        }

        public static Variable[] Invoke(Variable x, Axis[] axes = null)
        {
            return new Transpose(axes).BaseForward(Params<Variable>.args(x));
        }
    }
}
