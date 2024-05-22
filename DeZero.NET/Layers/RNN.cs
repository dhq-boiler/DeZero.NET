using DeZero.NET.Core;
using DeZero.NET.Functions;

namespace DeZero.NET.Layers
{
    public class RNN : Layer
    {
        public Property<Linear> x2h { get; private set; } = new();
        public Property<Linear> h2h { get; private set; } = new();
        public Property<Variable> h { get; private set; } = new();

        public RNN(int hidden_size, int? in_size = null) : base()
        {
            x2h.Value = new Layers.Linear(hidden_size, in_size: in_size);
            h2h.Value = new Layers.Linear(hidden_size, in_size: in_size, nobias: true);
            h = null;
        }

        public void ResetState()
        {
            h = null;
        }

        public override Variable[] Forward(Variable[] xs)
        {
            var x = xs[0];
            Variable h_new;
            if (h is null)
            {
                h_new = Tanh.Invoke(x)[0];
            }
            else
            {
                h_new = Tanh.Invoke(x2h.Value.Call(x)[0] + h2h.Value.Call(h.Value)[0])[0];
            }
            this.h.Value = h_new;
            return [h_new];
        }
    }
}
