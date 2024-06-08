using DeZero.NET.Core;

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
            int N = x.Shape[0], C = x.Shape[1], H = x.Shape[2], W = x.Shape[3];
            var out_h = (int)(1 + (H - KernelSize.Item2) / Stride.Item2);
            var out_w = (int)(1 + (W - KernelSize.Item1) / Stride.Item1);

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
            @out = Transpose.Invoke(Reshape.Invoke(@out, new Shape(N, out_h, out_w, C))[0], [new Axis([0, 3, 1, 2])])[0];
            return [@out];
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Get<Variable>(0);
            var pool_size = KernelSize.Item1 * KernelSize.Item2;
            using var dmax = xp.zeros(new Shape(gy.size, pool_size));
            using var first = xp.arange(ArgMax.size);
            using var second = ArgMax.flatten();
            dmax[first, second] = gy.Data.Value.flatten();
            using var dmax2 = dmax.reshape([..gy.Shape.Dimensions, pool_size]);
            using var dcol = dmax2.reshape(dmax2.shape[0] * dmax2.shape[1] * dmax2.shape[2], -1);
            var dx = Col2im.Invoke(dcol.ToVariable(), Inputs.ElementAt(0).NDarray.shape, KernelSize, Stride, Pad);
            return [dx];
        }

        public static Variable Invoke(Variable x, (int, int) kernelSize, (int, int) stride, (int, int) pad)
        {
            return new MaxPooling(kernelSize, stride, pad).Call(Params.New.SetPositionalArgs(x))[0];
        }
    }
}
