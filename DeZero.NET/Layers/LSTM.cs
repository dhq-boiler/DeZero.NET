using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeZero.NET.Layers
{
    public class LSTM : Layer
    {
        public Linear x2f { get; }
        public Linear x2i { get; }
        public Linear x2o { get; }
        public Linear x2u { get; }
        public Linear h2f { get; }
        public Linear h2i { get; }
        public Linear h2o { get; }
        public Linear h2u { get; }
        public Variable h { get; set; }
        public Variable c { get; set; }

        public LSTM(int hidden_size, int? in_size = null) : base()
        {
            int H = hidden_size;
            int I = in_size.Value;
            x2f = new Linear(hidden_size, in_size: I);
            x2i = new Linear(hidden_size, in_size: I);
            x2o = new Linear(hidden_size, in_size: I);
            x2u = new Linear(hidden_size, in_size: I);
            h2f = new Linear(hidden_size, in_size: H, nobias: true);
            h2i = new Linear(hidden_size, in_size: H, nobias: true);
            h2o = new Linear(hidden_size, in_size: H, nobias: true);
            h2u = new Linear(hidden_size, in_size: H, nobias: true);
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
                f = Functions.Sigmoid.Invoke(x2f.Call(x)[0])[0];
                i = Functions.Sigmoid.Invoke(x2i.Call(x)[0])[0];
                o = Functions.Sigmoid.Invoke(x2o.Call(x)[0])[0];
                u = Functions.Tanh.Invoke(x2u.Call(x)[0])[0];
            }
            else
            {
                f = Functions.Sigmoid.Invoke(x2f.Call(x)[0] + h2f.Call(h)[0])[0];
                i = Functions.Sigmoid.Invoke(x2i.Call(x)[0] + h2i.Call(h)[0])[0];
                o = Functions.Sigmoid.Invoke(x2o.Call(x)[0] + h2o.Call(h)[0])[0];
                u = Functions.Tanh.Invoke(x2u.Call(x)[0] + h2u.Call(h)[0])[0];
            }

            Variable c_new, h_new;
            if (c is null)
            {
                c_new = i * u;
            }
            else
            {
                c_new = (f * c) + (i * u);
            }

            h_new = o * Functions.Tanh.Invoke(c_new)[0];

            this.h = h_new;
            this.c = c_new;
            return [h_new];
        }
    }
}
