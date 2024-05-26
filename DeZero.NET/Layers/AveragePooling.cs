using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeZero.NET.Layers
{
    public class AveragePooling : Layer
    {
        public (int, int) KernelSize { get; }
        public int Stride { get; }
        public int Pad { get; }

        public AveragePooling((int, int) kernelSize, int stride, int pad)
        {
            this.KernelSize = kernelSize;
            this.Stride = stride;
            this.Pad = pad;
        }

        public override Variable[] Forward(params Variable[] xs)
        {
            var x = xs[0];
            return Functions.AveragePooling.Invoke(x, KernelSize, Stride, Pad);
        }
    }
}
