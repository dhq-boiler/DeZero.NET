using DeZero.NET.Core;

namespace DeZero.NET.Functions
{
    public class Deconv2d : Conv2d
    {
        public (int, int)? OutSize { get; set; }
        public bool no_bias { get; set; }

        public Deconv2d((int, int)? stride, (int, int)? pad, (int, int)? outsize = null) : base(stride.Value, pad.Value)
        {
            Stride = (1, 1);
            Pad = (0, 0);
            if (stride.HasValue)
            {
                Stride = stride.Value;
            }
            if (pad.HasValue)
            {
                Pad = pad.Value;
            }
            if (outsize.HasValue)
            {
                OutSize = outsize.Value;
            }
        }

        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>("x").Data.Value;
            var W = args.Get<Variable>("W");
            var b = args.Get<Variable>("b");

            var Weight = W.Data.Value;
            int SH = Stride.Item1, SW = Stride.Item2;
            int PH = Pad.Item1, PW = Pad.Item2;
            int C = Weight.shape[0], OC = Weight.shape[1], KH = Weight.shape[2], KW = Weight.shape[3];
            int N = x.shape[0], _ = x.shape[1], H = x.shape[2], _W = x.shape[3];

            int out_h, out_w;
            if (OutSize is null)
            {
                out_h = Utils.get_deconv_outsize(H, KH, SH, PH);
                out_w = Utils.get_deconv_outsize(_W, KW, SW, PW);
            }
            else
            {
                out_h = OutSize.Value.Item1;
                out_w = OutSize.Value.Item2;
            }

            var img_shape = (N, OC, out_h, out_w);

            var gcol = xp.tensordot(Weight, x, [0, 1]);
            gcol = xp.rollaxis(gcol, 3);
            var y = Utils.col2im_array(gcol, img_shape, (KH, KW), Stride, Pad, to_matrix: false);

            if (b is not null)
            {
                no_bias = true;
                y += b.Data.Value.reshape(new Shape(1, b.size, 1, 1));
            }

            return [y.ToVariable()];
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Get<Variable>(0);
            var x = Inputs.ElementAt(0).Variable;
            var W = Inputs.ElementAt(1).Variable;
            var b = Inputs.ElementAt(2).Variable;

            var gx = Conv2d.Invoke(gy, W, b: null, stride: Stride, pad: Pad);

            var f = new Conv2DGradW(this);
            var gW = f.Call(Params.New.SetPositionalArgs(gy, x));

            NDarray gb = null;
            if (b.Data.Value is not null)
            {
                gb = gy.Data.Value.sum(axis: new Axis([0, 2, 3]));
            }

            return [gx[0], gW[0], gb.ToVariable()];
        }   

        private (T, T) Pair<T>(T value)
        {
            return (value, value);
        }

        public static Variable[] Invoke(Variable x, Variable W, Variable b = null, (int, int)? stride = null, (int, int)? pad = null, (int, int)? outsize = null)
        {
            if (!stride.HasValue)
            {
                stride = (1, 1);
            }
            if (!pad.HasValue)
            {
                pad = (0, 0);
            }
            return new Deconv2d(stride, pad, outsize).Call(Params.New.SetKeywordArg(x, W, b));
        }
    }
}
