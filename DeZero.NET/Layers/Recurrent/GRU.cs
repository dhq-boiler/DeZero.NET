using DeZero.NET.Extensions;

namespace DeZero.NET.Layers.Recurrent
{
    public class GRU : Layer
    {
        private DeZero.NET.Layers.Linear.Linear Wxz, Wxr, Wxh, Whz, Whr, Whh;
        private Variable H;

        public GRU(int inSize, int hiddenSize)
        {
            Wxz = new DeZero.NET.Layers.Linear.Linear(in_size: inSize, out_size: hiddenSize);
            Wxr = new DeZero.NET.Layers.Linear.Linear(in_size: inSize, out_size: hiddenSize);
            Wxh = new DeZero.NET.Layers.Linear.Linear(in_size: inSize, out_size: hiddenSize);
            Whz = new DeZero.NET.Layers.Linear.Linear(in_size: hiddenSize, out_size: hiddenSize, nobias: true);
            Whr = new DeZero.NET.Layers.Linear.Linear(in_size: hiddenSize, out_size: hiddenSize, nobias: true);
            Whh = new DeZero.NET.Layers.Linear.Linear(in_size: hiddenSize, out_size: hiddenSize, nobias: true);
        }

        public override Variable[] Forward(params Variable[] variables)
        {
            var x = variables[0];
            Console.WriteLine($"Input x shape: {string.Join(", ", x.Shape)}");

            var batchSize = x.Shape[0];
            var seqLen = x.Shape[1];
            var inputSize = x.Shape[2];

            if (H == null || H.Shape[0] != batchSize)
            {
                H = xp.zeros(new Shape(batchSize, Wxz.OutSize.Value), dtype: Dtype.float32).ToVariable();
            }

            Console.WriteLine($"H shape: {string.Join(", ", H.Shape)}");

            var newH = new List<Variable>();

            for (int t = 0; t < seqLen; t++)
            {
                var xt = DeZero.NET.Functions.SliceFunc.Invoke(x, [new Slice(0, t, 0), new Slice(batchSize, t + 1, inputSize)])[0];
                xt = DeZero.NET.Functions.SliceFunc.Invoke(xt, [new Slice(1)])[0];  // Remove time dimension

                var wxz = Wxz.Forward(xt)[0];
                var whz = Whz.Forward(H)[0];
                var z = DeZero.NET.Functions.Sigmoid.Invoke(DeZero.NET.Functions.Add.Invoke(wxz, whz).Item1[0])[0];

                var wxr = Wxr.Forward(xt)[0];
                var whr = Whr.Forward(H)[0];
                var r = DeZero.NET.Functions.Sigmoid.Invoke(DeZero.NET.Functions.Add.Invoke(wxr, whr).Item1[0])[0];

                var wxh = Wxh.Forward(xt)[0];
                var whh = Whh.Forward(DeZero.NET.Functions.Mul.Invoke(r, H)[0])[0];
                var h_tilde = DeZero.NET.Functions.Tanh.Invoke(DeZero.NET.Functions.Add.Invoke(wxh, whh).Item1[0])[0];

                H = DeZero.NET.Functions.Add.Invoke(
                    DeZero.NET.Functions.Mul.Invoke(z, H)[0],
                    DeZero.NET.Functions.Mul.Invoke(DeZero.NET.Functions.Sub.Invoke(xp.array(1).ToVariable(), z)[0], h_tilde)[0]
                ).Item1[0];

                newH.Add(H);
            }

            // Stack all hidden states
            H = DeZero.NET.Functions.Stack.Invoke(newH.ToArray(), axis: 1)[0];

            Console.WriteLine($"Output H shape: {string.Join(", ", H.Shape)}");
            return [H];
        }

        public void ResetState()
        {
            H = null;
        }
    }
}
