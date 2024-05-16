using DeZero.NET.Functions;

namespace DeZero.NET.Layers
{
    public class RNN : Layer
    {
        public Linear x2h { get; private set; }
        public Linear h2h { get; private set; }
        public Variable h { get; private set; }

        public RNN(int hidden_size, int? in_size = null) : base()
        {
            x2h = new Layers.Linear(hidden_size, in_size: in_size);
            h2h = new Layers.Linear(hidden_size, in_size: in_size, nobias: true);
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
                h_new = Tanh.Invoke(x2h.Call(x)[0] + h2h.Call(h)[0])[0];
            }
            this.h = h_new;
            return [h_new];
        }
    }
}
