using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeZero.NET.Layers
{
    public class Linear : Layer
    {
        public Parameter b { get; set; }
        public Parameter W { get; set; }
        public int OutSize { get; }
        public Dtype Dtype { get; }
        public int? InSize { get; set; }

        public override Func<Variable[], Variable[]> F => xs => Forward(xs);

        public Linear(int out_size, string dtype = "f8", bool nobias = false, int? in_size = null)
            : this(out_size, new Dtype(dtype), nobias, in_size)
        {
        }

        public Linear(int out_size, Dtype dtype, bool nobias = false, int? in_size = null)
        {
            OutSize = out_size;
            Dtype = dtype;
            InSize = in_size;

            W = new Parameter(null, name: "W");

            if (InSize is not null)
            {
                _init_W();
            }

            if (nobias)
            {
                b = null;
            }
            else
            {
                b = new Parameter(xp.zeros(OutSize, dtype: dtype).ToVariable(), name: "b");
            }
        }


        private void _init_W()
        {
            int I = InSize.Value, O = OutSize;
            var W_data = xp.random.randn(I, O).astype(Dtype) * xp.sqrt(new NDarray(1f / I)).asscalar<float>();
            W.Data = W_data;
        }

        public override Variable[] Forward(params Variable[] xs)
        {
            var x = xs[0];
            if (W.Data is null)
            {
                InSize = x.Shape[1];
                _init_W();
            }

            var ys = Functions.Linear.Invoke(x, W, b);
            return ys;
        }
    }
}
