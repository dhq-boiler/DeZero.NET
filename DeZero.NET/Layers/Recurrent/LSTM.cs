using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DeZero.NET.Core;

namespace DeZero.NET.Layers.Recurrent
{
    /// <summary>
    /// Long Short-Term Memory (LSTM) layer
    /// ゲート付きリカレントニューラルネットワークのLSTMレイヤ
    /// </summary>
    public class LSTM : Layer
    {
        public Property<Linear.Linear> x2f { get; } = new(nameof(x2f));
        public Property<Linear.Linear> x2i { get; } = new(nameof(x2i));
        public Property<Linear.Linear> x2o { get; } = new(nameof(x2o));
        public Property<Linear.Linear> x2u { get; } = new(nameof(x2u));
        public Property<Linear.Linear> h2f { get; } = new(nameof(h2f));
        public Property<Linear.Linear> h2i { get; } = new(nameof(h2i));
        public Property<Linear.Linear> h2o { get; } = new(nameof(h2o));
        public Property<Linear.Linear> h2u { get; } = new(nameof(h2u));
        public Property<Variable> h { get; set; } = new(nameof(h));
        public Property<Variable> c { get; set; } = new(nameof(c));

        public LSTM() : base()
        {
            RegisterEvent(x2f, x2i, x2o, x2u, h2f, h2i, h2o, h2u, h, c);
            ResetState();
        }

        public LSTM(int hidden_size, int? in_size = null) : this()
        {
            int H = hidden_size;
            int I = in_size.Value;
            x2f.Value = new Linear.Linear(hidden_size, in_size: I);
            x2i.Value = new Linear.Linear(hidden_size, in_size: I);
            x2o.Value = new Linear.Linear(hidden_size, in_size: I);
            x2u.Value = new Linear.Linear(hidden_size, in_size: I);
            h2f.Value = new Linear.Linear(hidden_size, in_size: H, nobias: true);
            h2i.Value = new Linear.Linear(hidden_size, in_size: H, nobias: true);
            h2o.Value = new Linear.Linear(hidden_size, in_size: H, nobias: true);
            h2u.Value = new Linear.Linear(hidden_size, in_size: H, nobias: true);
            ResetState();
        }

        private void ResetState()
        {
            h.Value = null;
            c.Value = null;
        }

        public override Variable[] Forward(params Variable[] xs)
        {
            var x = xs[0];
            Variable f, i, o, u;
            if (h is null)
            {
                f = Functions.Sigmoid.Invoke(x2f.Value.Call(x)[0])[0];
                i = Functions.Sigmoid.Invoke(x2i.Value.Call(x)[0])[0];
                o = Functions.Sigmoid.Invoke(x2o.Value.Call(x)[0])[0];
                u = Functions.Tanh.Invoke(x2u.Value.Call(x)[0])[0];
            }
            else
            {
                f = Functions.Sigmoid.Invoke(x2f.Value.Call(x)[0] + h2f.Value.Call(h.Value)[0])[0];
                i = Functions.Sigmoid.Invoke(x2i.Value.Call(x)[0] + h2i.Value.Call(h.Value)[0])[0];
                o = Functions.Sigmoid.Invoke(x2o.Value.Call(x)[0] + h2o.Value.Call(h.Value)[0])[0];
                u = Functions.Tanh.Invoke(x2u.Value.Call(x)[0] + h2u.Value.Call(h.Value)[0])[0];
            }

            Variable c_new, h_new;
            if (c is null)
            {
                c_new = i * u;
            }
            else
            {
                c_new = f * c.Value + i * u;
            }

            h_new = o * Functions.Tanh.Invoke(c_new)[0];

            h.Value = h_new;
            c.Value = c_new;
            return [h_new];
        }
    }
}
