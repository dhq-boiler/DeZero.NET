using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class Col2im : Function
    {
        public Shape input_shape { get; set; }
        public (int, int) KernelSize { get; set; }
        public (int, int) Pad { get; set; }
        public bool ToMatrix { get; set; }

        public (int, int) Stride { get; set; }
        public Col2im(Shape input_shape, (int, int) kernelSize, (int, int) stride, (int, int) pad,
            bool toMatrix) : base()
        {
            this.input_shape = input_shape;
            KernelSize = kernelSize;
            Stride = stride;
            Pad = pad;
            ToMatrix = toMatrix;
        }

        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>("x");
            var input_shape_tuple = (input_shape.Dimensions[0], input_shape.Dimensions[1], input_shape.Dimensions[2], input_shape.Dimensions[3]);
            var y = Utils.col2im_array(x.Data.Value, input_shape_tuple, KernelSize, Stride, Pad, ToMatrix);
            return [y.Relay(this)];
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Get<Variable>(0);
            var gx = Im2col.Invoke(gy, KernelSize, Stride, Pad, ToMatrix);
            return [gx];
        }

        public static Variable Invoke(Variable x, Shape input_shape, (int, int) kernelSize, (int, int)? stride,
            (int, int)? pad,
            bool toMatrix = true)
        {
            if (!stride.HasValue)
            {
                stride = (1, 1);
            }

            if (!pad.HasValue)
            {
                pad = (0, 0);
            }
            return new Col2im(input_shape, kernelSize, stride.Value, pad.Value, toMatrix).Call(Params.New.SetPositionalArgs(x))[0];
        }
    }
}
