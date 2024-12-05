using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class Im2col : Function
    {
        public (int KH, int KW) KernelSize { get; }
        public (int, int) Stride { get; }
        public (int, int) Pad { get; }
        public bool ToMatrix { get; }
        public Shape input_shape { get; set; }

        public Im2col((int KH, int KW) kernelSize, (int, int) stride, (int, int) pad, bool toMatrix) : base()
        {
            KernelSize = kernelSize;
            Stride = stride;
            Pad = pad;
            ToMatrix = toMatrix;
        }

        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>("x");
            this.input_shape = x.Shape;
            var y = Utils.im2col_array(x, KernelSize, Stride, Pad, ToMatrix);
            return [y.Relay(this)];
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Get<Variable>(0);
            var gx = Col2im.Invoke(gy, input_shape, KernelSize, Stride, Pad, ToMatrix);
            return [gx];
        }

        public static Variable Invoke(Variable x, (int KH, int KW) kernelSize, (int, int)? stride, (int, int)? pad,
            bool toMatrix=true)
        {
            if (!stride.HasValue)
            {
                stride = (1, 1);
            }

            if (!pad.HasValue)
            {
                pad = (0, 0);
            }
            return new Im2col(kernelSize, stride.Value, pad.Value, toMatrix).Call(Params.New.SetPositionalArgs(x))[0];
        }
    }
}
