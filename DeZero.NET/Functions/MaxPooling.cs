using DeZero.NET.Core;
using DeZero.NET.Extensions;
using System.Diagnostics;

namespace DeZero.NET.Functions
{
    public class MaxPooling : Function
    {
        public (int, int) KernelSize { get; set; }
        public (int, int) Stride { get; set; }
        public (int, int) Pad { get; set; }
        public NDarray ArgMax { get; set; }

        public MaxPooling((int, int) kernelSize, (int, int) stride, (int, int) pad)
        {
            this.KernelSize = kernelSize;
            this.Stride = stride;
            this.Pad = pad;
        }

        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>(0);
            if (x.ndim != 4)
            {
                throw new ArgumentException("入力は4次元テンソルである必要があります。", nameof(x));
            }
            using var x_shape = x.Shape;
            int N = x_shape[0], C = x_shape[1], H = x_shape[2], W = x_shape[3];
            var out_h = (int)(1 + (H + 2 * Pad.Item2 - KernelSize.Item2) / Stride.Item2);
            var out_w = (int)(1 + (W + 2 * Pad.Item1 - KernelSize.Item1) / Stride.Item1);

            using var col = Im2col.Invoke(x, KernelSize, Stride, Pad);
            using var col2 = Reshape.Invoke(col, new Shape(-1, KernelSize.Item1 * KernelSize.Item2))[0];
            var arg_max = xp.argmax(col2.Data.Value, axis:1);
            var @out = xp.max(col2.Data.Value, axis:[1]).ToVariable(this);
            if (this.ArgMax is not null)
            {
                this.ArgMax?.Dispose();
                this.ArgMax = null;
            }
            this.ArgMax = arg_max;
            using var shape1 = new Shape(N, C, out_h, out_w);
            @out = Reshape.Invoke(@out, shape1)[0];
            using var @out_shape = @out.Shape;
            Debug.Assert(@out.ndim == 4, "出力は4次元テンソルである必要があります。");
            Debug.Assert(@out_shape[0] == N && @out_shape[1] == C, "バッチサイズとチャンネル数は保持されるべきです。");
            return [@out.Relay(this)];
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Get<Variable>(0);
            var pool_size = KernelSize.Item1 * KernelSize.Item2;
            using var zeros_shape = new Shape(gy.Data.Value.size, pool_size);
            using var dmax = xp.zeros(zeros_shape);
            using var first = xp.arange(ArgMax.size);
            using var second = ArgMax.flatten();
            dmax[first, second] = gy.Data.Value.flatten();
            using var gy_shape = gy.Shape;
            using var dmax2 = dmax.reshape([.. gy_shape.Dimensions, pool_size]);
            using var dmax2_shape = dmax2.shape;
            using var dcol = dmax2.reshape(dmax2_shape[0] * dmax2_shape[1] * dmax2_shape[2], -1);
            using var input_0_shape = Inputs.ElementAt(0).Variable.Shape;
            var dx = Col2im.Invoke(dcol.ToVariable(), input_0_shape, KernelSize, Stride, Pad);
            return [dx];
        }

        public static Variable[] Invoke(Variable x, (int, int) kernelSize, (int, int) stride, (int, int) pad)
        {
            return new MaxPooling(kernelSize, stride, pad).Call(Params.New.SetPositionalArgs(x));
        }
    }
}
