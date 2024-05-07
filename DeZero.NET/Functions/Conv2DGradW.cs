using DeZero.NET.Core;

namespace DeZero.NET.Functions
{
    public class Conv2DGradW : Function
    {
        public (int, int) kernel_size { get; set; }
        public (int, int) stride { get; set; }
        public (int, int) pad { get; set; }

        public Conv2DGradW(Conv2d conv2d) : base()
        {
            var W = conv2d.Inputs.ElementAt(1).Variable;
            int kh = W.Shape[3], kw = W.Shape[4];
            this.kernel_size = (kh, kw);
            this.stride = conv2d.Stride;
            this.pad = conv2d.Pad;
        }

        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>("x");
            var gy = args.Get<Variable>("gy");
            var col = Utils.im2col_array(x, kernel_size, stride, pad, to_matrix: false);
            var gW = xp.tensordot(gy.Data, col.Data, [[0, 2, 3], [0, 4, 5]]);
            return [gW.ToVariable()];
        }

        public override Variable[] Backward(Params args)
        {
            var gys = args.Get<Variable>(0);
            var x = Inputs.ElementAt(0).Variable;
            var gy = Inputs.ElementAt(1).Variable;
            var gW = Outputs.ElementAt(0);

            int xh = x.Shape[2], xw = x.Shape[3];
            var gx = Deconv2d.Invoke(gy, gW, stride: stride, pad: pad, outsize: (xh, xw));
            var ggy = Conv2d.Invoke(x, gW, stride: stride, pad: pad);
            return [gx[0], ggy[0]];
        }
    }
}
