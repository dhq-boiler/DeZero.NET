using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DeZero.NET.Core;

namespace DeZero.NET.Layers
{
    public class LSTM : Layer
    {
        public Property<Linear> x2f { get; } = new();
        public Property<Linear> x2i { get; } = new();
        public Property<Linear> x2o { get; } = new();
        public Property<Linear> x2u { get; } = new();
        public Property<Linear> h2f { get; } = new();
        public Property<Linear> h2i { get; } = new();
        public Property<Linear> h2o { get; } = new();
        public Property<Linear> h2u { get; } = new();
        public Property<Variable> h { get; set; } = new();
        public Property<Variable> c { get; set; } = new();

        public LSTM(int hidden_size, int? in_size = null) : base()
        {
            int H = hidden_size;
            int I = in_size.Value;
            x2f.Value = new Linear(hidden_size, in_size: I);
            x2i.Value = new Linear(hidden_size, in_size: I);
            x2o.Value = new Linear(hidden_size, in_size: I);
            x2u.Value = new Linear(hidden_size, in_size: I);
            h2f.Value = new Linear(hidden_size, in_size: H, nobias: true);
            h2i.Value = new Linear(hidden_size, in_size: H, nobias: true);
            h2o.Value = new Linear(hidden_size, in_size: H, nobias: true);
            h2u.Value = new Linear(hidden_size, in_size: H, nobias: true);
            ResetState();
        }

        private void ResetState()
        {
            h = null;
            c = null;
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
                c_new = (f * c.Value) + (i * u);
            }

            h_new = o * Functions.Tanh.Invoke(c_new)[0];

            this.h.Value = h_new;
            this.c.Value = c_new;
            return [h_new];
        }
    }
}
