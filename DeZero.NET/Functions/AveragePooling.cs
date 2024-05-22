using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DeZero.NET.Core;

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
            var col = Utils.im2col_array(x, KernelSize, (Stride, Stride), (Pad, Pad), to_matrix: false);
            var y = col.Data.Value.mean(axis: new Axis([2, 3])).ToVariable(this);
            return [y];
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Get<Variable>(0);
            int N = gy.Shape[0], C = gy.Shape[1], OH = gy.Shape[2], OW = gy.Shape[3];
            int KH = KernelSize.Item1, KW = KernelSize.Item2;
            gy /= (KW * KH);
            var gcol = gy.reshape(-1)[0].Data.Value.broadcast_to(new Shape(KH, KW, N * C * OH * OW)).ToVariable();
            gcol = gcol.reshape(KH, KW, N, C, OH, OW)[0].transpose(2, 3, 0, 1, 4, 5)[0];
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
