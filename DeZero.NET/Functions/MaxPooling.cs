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

            var col = Im2col.Invoke(x, KernelSize, Stride, Pad);
            col = Reshape.Invoke(col, new Shape(-1, KernelSize.Item1 * KernelSize.Item2))[0];
            var arg_max = xp.argmax(col.Data.Value, axis:1);
            var @out = xp.max(col.Data.Value, axis:[1]).ToVariable(this);
            this.ArgMax = arg_max;
            @out = Transpose.Invoke(Reshape.Invoke(@out, new Shape(N, out_h, out_w, C))[0], [new Axis([0, 3, 1, 2])])[0];
            return [@out];
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Get<Variable>(0);
            var pool_size = KernelSize.Item1 * KernelSize.Item2;
            var dmax = xp.zeros(new Shape(gy.size, pool_size));
            dmax[xp.arange(ArgMax.size), ArgMax.flatten()] = gy.Data.Value.flatten();
            dmax = dmax.reshape([..gy.Shape.Dimensions, pool_size]);
            var dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1);
            var dx = Col2im.Invoke(dcol.ToVariable(), Inputs.ElementAt(0).NDarray.shape, KernelSize, Stride, Pad);
            return [dx];
        }

        public static Variable Invoke(Variable x, (int, int) kernelSize, (int, int) stride, (int, int) pad)
        {
            return new MaxPooling(kernelSize, stride, pad).Call(Params.New.SetPositionalArgs(x))[0];
        }
    }
}
