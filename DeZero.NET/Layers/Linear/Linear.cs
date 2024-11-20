using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Layers.Linear
{
    /// <summary>
    /// 総結合層
    /// </summary>
    public class Linear : Layer, IWbOwner
    {
        public Property<Parameter> b { get; } = new(nameof(b));
        public Property<Parameter> W { get; } = new(nameof(W));
        public Property<int> OutSize { get; } = new(nameof(OutSize));
        public Property<Dtype> Dtype { get; } = new(nameof(Dtype));
        public Property<int?> InSize { get; } = new(nameof(InSize));
        public override Func<Variable[], Variable[]> F => xs => Forward(xs);
        public Action WInitialized { get; set; }
        public bool NoBias => b.Value is null;

        public Linear()
        {
            RegisterEvent(b, W, OutSize, Dtype, InSize);
        }

        public Linear(int out_size, Dtype dtype, bool nobias = false, int? in_size = null) : this()
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

        public Linear(int out_size, string dtype = "f8", bool nobias = false, int? in_size = null)
            : this(out_size, new Dtype(dtype), nobias, in_size)
        {
        }

        private void _init_W()
        {
            int I = InSize.Value.Value, O = OutSize.Value;
            using var random = xp.random.randn(I, O);
            using var sqrt = xp.sqrt(new NDarray(1f / I));
            var W_data = random.astype(Dtype.Value) * sqrt.asscalar<float>();
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

            WInitialized?.Invoke();

            var ys = Functions.Linear.Invoke(x, W.Value, b.Value);
            return ys;
        }
    }
}
