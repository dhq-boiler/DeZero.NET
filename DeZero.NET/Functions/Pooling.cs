using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DeZero.NET.Core;

namespace DeZero.NET.Functions
{
    public class Pooling : Function
    {
        public (int, int) KernelSize { get; set; }
        public int Stride { get; set; }
        public int Pad { get; set; }
        public NDarray Indexes { get; set; }

        public Pooling((int, int) kernelSize, int stride = 1, int pad = 0)
        {
            KernelSize = kernelSize;
            Stride = stride;
            Pad = pad;
        }

        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>(0);
            var col = Utils.im2col_array(x, KernelSize, (Stride, Stride), (Pad, Pad), to_matrix: false);

            int N = col.Shape[0], C = col.Shape[1], KH = col.Shape[2], KW = col.Shape[3], OH = col.Shape[4], OW = col.Shape[5];
            col = col.reshape(N, C, KH * KW, OH, OW)[0];
            Indexes = col.Data.Value.argmax(axis: 2);
            var y = col.Data.Value.max(axis: [2]);
            return [y.ToVariable()];
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Get<Variable>(0);
            return new Pooling2DGrad(this).Call(Params.New.SetPositionalArgs(gy));
        }

        public static Variable[] Invoke(Variable x, (int, int) kernelSize, int stride = 1, int pad = 0)
        {
            return new Pooling(kernelSize, stride, pad).Call(Params.New.SetPositionalArgs(x));
        }
    }
}
