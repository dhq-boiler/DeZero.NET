using DeZero.NET.Core;

namespace DeZero.NET.Functions
{
    public class Conv2d : Function
    {
        public (int, int) Stride { get; set; }
        public (int, int) Pad { get; set; }

        public Conv2d(int stride, int pad) : this((stride, stride), (pad, pad))
        {
        }

        public Conv2d((int, int) stride, (int, int) pad) : base()
        {
            Stride = stride;
            Pad = pad;
        }

        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>("x");
            var W = args.Get<Variable>("W");
            var b = args.Get<Variable>("b");

            Shape KH = W.Shape[2], KW = W.Shape[3];
            var col = Utils.im2col_array(x, (KH[0], KW[0]), Stride, Pad, to_matrix:false);

            var y = xp.tensordot(col.Data, W.Data, [[1, 2, 3], [1, 2, 3]]);
            if (b is not null)
            {
                y += b.Data;
            }

            y = xp.rollaxis(y, 3, 1);

            return [y.ToVariable()];
        }

        public override Variable[] Backward(Params args)
        {
            var x = Inputs.ElementAt(0).Variable;
            var W = Inputs.ElementAt(1).Variable;
            var b = Inputs.ElementAt(2).Variable;
            var gy = args.Get<Variable>(0);

            var gx = Deconv2d.Invoke(gy, W, b: null, stride: Stride, pad: Pad, outsize: (x.Shape[2], x: x.Shape[3]));
            var gW = new Conv2DGradW(this).Call(Params.New.SetPositionalArgs(x, gy));
            NDarray gb = null;
            if (b.Data is not null)
            {
                gb = gy.Data.sum(axis: new Axis([0, 2, 3]));
            }
            return [gx[0], gW[0], gb.ToVariable()];
        }   

        private (T, T) Pair<T>(T value)
        {
            return (value, value);
        }

        public static Variable[] Invoke(Variable x, Variable W, Variable b = null, (int, int)? stride = null, (int, int)? pad = null)
        {
            if (!stride.HasValue)
            {
                stride = (1, 1);
            }
            if (!pad.HasValue)
            {
                pad = (0, 0);
            }
            return new Conv2d(stride.Value, pad.Value).Call(Params.New.SetKeywordArg(x, W, b));
        }

        public static Variable[] Invoke(Variable x, Variable W, Variable b = null, int stride = 1, int pad = 0)
        {
            return new Conv2d((stride, stride), (pad, pad)).Call(Params.New.SetKeywordArg(x, W, b));
        }
    }
}
