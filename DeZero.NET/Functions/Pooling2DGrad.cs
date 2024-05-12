using System.Runtime;
using Cupy;
using DeZero.NET.Core;

namespace DeZero.NET.Functions
{
    internal class Pooling2DGrad : Function
    {
        private Pooling mpool2d;

        public (int, int) KernelSize { get; set; }
        public int Stride { get; set; }
        public int Pad { get; set; }
        public Shape InputShape { get; set; }
        public Dtype Dtype { get; set; }
        public NDarray Indexes { get; set; }

        public Pooling2DGrad(Pooling mpool2d)
        {
            this.mpool2d = mpool2d;
            this.KernelSize = mpool2d.KernelSize;
            this.Stride = mpool2d.Stride;
            this.Pad = mpool2d.Pad;
            this.InputShape = mpool2d.Inputs.ElementAt(0).Variable.Shape;
            this.Dtype = mpool2d.Inputs.ElementAt(0).Variable.Dtype;
            this.Indexes = mpool2d.Indexes;
        }

        public override Variable[] Forward(Params args)
        {
            var gy = args.Get<Variable>(0);
            int N = gy.Shape[0], C = gy.Shape[1], OH = gy.Shape[2], OW = gy.Shape[3];
            N = InputShape[0];
            C = InputShape[1];
            int H = InputShape[2], W = InputShape[3];
            int KH = KernelSize.Item1, KW = KernelSize.Item2;

            var gcol = xp.zeros(N * C * OH * OW * KH * KW, dtype: Dtype);

            var indexes = Indexes.ravel() + xp.arange(0, Indexes.size * KH * KW, KH * KW);

            gcol[indexes] = gy.Data.ravel();
            gcol = gcol.reshape(N, C, OH, OW, KH, KW);
            gcol = xp.swapaxes(gcol, 2, 4);
            gcol = xp.swapaxes(gcol, 3, 5);

            var gx = Utils.col2im_array(gcol, (N, C, H, W), KernelSize, (Stride, Stride), (Pad, Pad), to_matrix: false);
            return [gx.ToVariable(this)];
        }

        public override Variable[] Backward(Params args)
        {
            var ggx = args.Get<Variable>(0);
            var f = new Pooling2DWithIndexes(mpool2d);
            return f.Call(Params.New.SetPositionalArgs(ggx));
        }
    }
}