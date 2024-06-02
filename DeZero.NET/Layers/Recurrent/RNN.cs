using DeZero.NET.Core;
using DeZero.NET.Layers.Linear;

namespace DeZero.NET.Layers.Recurrent
{
    /// <summary>
    /// 回帰結合層（リカレントニューラルネットワーク層）
    /// </summary>
    public class RNN : Layer
    {
        public Property<DeZero.NET.Layers.Linear.Linear> x2h { get; private set; } = new(nameof(x2h));
        public Property<Linear.Linear> h2h { get; private set; } = new(nameof(h2h));
        public Property<Variable> h { get; private set; } = new(nameof(h));

        public RNN() : base()
        {
            RegisterEvent(x2h, h2h, h);
        }

        public RNN(int hidden_size, int? in_size = null) : this()
        {
            x2h.Value = new Linear.Linear(hidden_size, in_size: in_size);
            h2h.Value = new Linear.Linear(hidden_size, in_size: in_size, nobias: true);
            h = null;
        }

        public void ResetState()
        {
            h.Value = null;
        }

        public override Variable[] Forward(Variable[] xs)
        {
            var x = xs[0];
            Variable h_new;
            if (h is null)
            {
                h_new = Functions.Tanh.Invoke(x)[0];
            }
            else
            {
                h_new = Functions.Tanh.Invoke(x2h.Value.Call(x)[0] + h2h.Value.Call(h.Value)[0])[0];
            }
            h.Value = h_new;
            return [h_new];
        }
    }
}
