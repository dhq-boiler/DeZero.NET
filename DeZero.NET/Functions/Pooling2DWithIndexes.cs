using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class Pooling2DWithIndexes : Function
    {
        public (int, int) KernelSize { get; set; }
        public int Stride { get; set; }
        public int Pad { get; set; }
        public Shape InputShape { get; set; }
        public Dtype Dtype { get; set; }
        public NDarray Indexes { get; set; }

        public Pooling2DWithIndexes(Pooling mpool2d)
        {
            this.KernelSize = mpool2d.KernelSize;
            this.Stride = mpool2d.Stride;
            this.Pad = mpool2d.Pad;
            this.InputShape = mpool2d.Inputs.ElementAt(0).Variable.Shape;
            this.Dtype = mpool2d.Inputs.ElementAt(0).Variable.Dtype;
            this.Indexes = mpool2d.Indexes;
        }

        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>(0);
            var col = Utils.im2col_array(x, KernelSize, (Stride, Stride), (Pad, Pad), to_matrix: false);
            int N = col.Shape[0], C = col.Shape[1], KH = col.Shape[2], KW = col.Shape[3], OH = col.Shape[4], OW = col.Shape[5];
            col = col.reshape(new Shape(N, C, KH * KW, OH, OW))[0];
            col = col.transpose(0, 1, 3, 4, 2)[0].reshape(new Shape(-1, KH * KW))[0];
            var indexes = Indexes.ravel();
            col = col.Data.Value[xp.arange(indexes.len), indexes].ToVariable();
            return [col.reshape(new Shape(N, C, OH, OW))[0].Relay(this)];
        }
    }
}
