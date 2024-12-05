using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class AveragePooling : Function
    {
        public (int, int) KernelSize { get; set; }
        public int Stride { get; set; }
        public int Pad { get; set; }
        public Shape InputShape { get; set; }
        public Func<Params, Variable[]> F { get; set; }

        public AveragePooling((int, int) kernelSize, int stride = 1, int pad = 0)
        {
            KernelSize = kernelSize;
            Stride = stride;
            Pad = pad;
            InputShape = null;
        }

        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>(0);
            InputShape = x.Shape;
            using var col = Utils.im2col_array(x, KernelSize, (Stride, Stride), (Pad, Pad), to_matrix: false).Relay(null, x);
            var y = col.Data.Value.mean(axis: new Axis([2, 3])).Relay(this, col);
            return [y];
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Get<Variable>(0);

            // Get dimensions
            int N = gy.Shape[0];  // batch size
            int C = gy.Shape[1];  // channels
            int OH = gy.Shape[2]; // output height
            int OW = gy.Shape[3]; // output width
            int KH = KernelSize.Item1;
            int KW = KernelSize.Item2;

            // Average gradient by kernel size
            gy /= (KW * KH);

            // Reshape and expand gradient to match kernel dimensions
            // First reshape gy to (N, C, OH, OW)
            using var gy_reshaped = gy.reshape(N, C, OH, OW)[0];

            // Expand dimensions to match kernel size
            using var gy_expanded = gy_reshaped.reshape(N, C, 1, 1, OH, OW)[0];
            using var gy_tiled = gy_expanded.Data.Value.broadcast_to(new Shape(N, C, KH, KW, OH, OW));

            // Transpose to get correct dimension order for col2im
            using var gcol = gy_tiled.transpose(0, 1, 2, 3, 4, 5).ToVariable();

            // Convert back to image format
            var gx = Col2im.Invoke(gcol, InputShape, KernelSize, (Stride, Stride), (Pad, Pad), toMatrix: false);

            return [gx];
        }

        public static Variable[] Invoke(Variable x, (int, int) kernelSize, int stride = 1, int pad = 0)
        {
            return new AveragePooling(kernelSize, stride, pad).Call(Params.New.SetPositionalArgs(x));
        }

        public static Variable[] Invoke(AveragePooling f, Variable x)
        {
            return f.F.Invoke(Params.New.SetPositionalArgs(x));
        }
    }
}
