using DeZero.NET.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeZero.NET.Layers
{
    public class Linear : Layer
    {
        public Property<Parameter> b { get; } = new();
        public Property<Parameter> W { get; } = new();
        public Property<int> OutSize { get; } = new();
        public Property<Dtype> Dtype { get; } = new();
        public Property<int?> InSize { get; } = new();

        public override Func<Variable[], Variable[]> F => xs => Forward(xs);

        public Linear(int out_size, string dtype = "f8", bool nobias = false, int? in_size = null)
            : this(out_size, new Dtype(dtype), nobias, in_size)
        {
        }

        public Linear(int out_size, Dtype dtype, bool nobias = false, int? in_size = null)
        {
            OutSize.Value = out_size;
            Dtype.Value = dtype;
            InSize.Value = in_size;

            W.Value = new Parameter(null, name: "W");

            if (InSize.Value is not null)
            {
                _init_W();
            }

            if (nobias)
            {
                b.Value = null;
            }
            else
            {
                b.Value = new Parameter(xp.zeros(OutSize.Value, dtype: dtype).ToVariable(), name: "b");
            }
        }


        private void _init_W()
        {
            int I = InSize.Value.Value, O = OutSize.Value;
            var W_data = xp.random.randn(I, O).astype(Dtype.Value) * xp.sqrt(new NDarray(1f / I)).asscalar<float>();
            W.Value.Data.Value = W_data;
        }

        public override Variable[] Forward(params Variable[] xs)
        {
            var x = xs[0];
            if (W.Value.Data.Value is null)
            {
                InSize.Value = x.Shape[1];
                _init_W();
            }

            var ys = Functions.Linear.Invoke(x, W.Value, b.Value);
            return ys;
        }
    }
}
