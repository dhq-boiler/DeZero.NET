using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Layers.Recurrent
{
    public class GRU : Layer
    {
        public Property<DeZero.NET.Layers.Linear.Linear> Wxz { get; } = new(nameof(Wxz)); 
        public Property<DeZero.NET.Layers.Linear.Linear> Wxr { get; } = new(nameof(Wxr));  
        public Property<DeZero.NET.Layers.Linear.Linear> Wxh { get; } = new(nameof(Wxh));  
        public Property<DeZero.NET.Layers.Linear.Linear> Whz { get; } = new(nameof(Whz)); 
        public Property<DeZero.NET.Layers.Linear.Linear> Whr { get; } = new(nameof(Whr));
        public Property<DeZero.NET.Layers.Linear.Linear> Whh { get; } = new(nameof(Whh));
        public Property<Variable> H { get; } = new(nameof(H));

        public GRU(int inSize, int hiddenSize)
        {
            RegisterEvent(Wxz, Wxr, Wxh, Whz, Whr, Whh, H);

            Wxz.Value = new DeZero.NET.Layers.Linear.Linear(in_size: inSize, out_size: hiddenSize);
            SetAttribute("Wxz", Wxz.Value);
            Wxr.Value = new DeZero.NET.Layers.Linear.Linear(in_size: inSize, out_size: hiddenSize);
            SetAttribute("Wxr", Wxr.Value);
            Wxh.Value = new DeZero.NET.Layers.Linear.Linear(in_size: inSize, out_size: hiddenSize);
            SetAttribute("Wxh", Wxh.Value);
            Whz.Value = new DeZero.NET.Layers.Linear.Linear(in_size: hiddenSize, out_size: hiddenSize, nobias: true);
            SetAttribute("Whz", Whz.Value);
            Whr.Value = new DeZero.NET.Layers.Linear.Linear(in_size: hiddenSize, out_size: hiddenSize, nobias: true);
            SetAttribute("Whr", Whr.Value);
            Whh.Value = new DeZero.NET.Layers.Linear.Linear(in_size: hiddenSize, out_size: hiddenSize, nobias: true);
            SetAttribute("Whh", Whh.Value);
        }

        public override Variable[] Forward(params Variable[] variables)
        {
            var x = variables[0];
            //Console.WriteLine($"Input x shape: {string.Join(", ", x.Shape)}");
            var batchSize = x.Shape[0];
            var inputSize = x.Shape[1];

            if (H.Value == null || H.Value.Shape[0] != batchSize)
            {
                H.Value = xp.zeros(new Shape(batchSize, Wxz.Value.OutSize.Value), dtype: Dtype.float32).ToVariable();
            }
            //Console.WriteLine($"H shape: {string.Join(", ", H.Shape)}");

            //Console.WriteLine($"x shape: {string.Join(", ", x.Shape)}");
            var wxz = Wxz.Value.Forward(x)[0];
            var whz = Whz.Value.Forward(H.Value)[0];
            var z = DeZero.NET.Functions.Sigmoid.Invoke(DeZero.NET.Functions.Add.Invoke(wxz, whz).Item1[0])[0];

            var wxr = Wxr.Value.Forward(x)[0];
            var whr = Whr.Value.Forward(H.Value)[0];
            var r = DeZero.NET.Functions.Sigmoid.Invoke(DeZero.NET.Functions.Add.Invoke(wxr, whr).Item1[0])[0];

            var wxh = Wxh.Value.Forward(x)[0];
            var whh = Whh.Value.Forward(DeZero.NET.Functions.Mul.Invoke(r, H.Value)[0])[0];
            var h_tilde = DeZero.NET.Functions.Tanh.Invoke(DeZero.NET.Functions.Add.Invoke(wxh, whh).Item1[0])[0];

            H.Value = DeZero.NET.Functions.Add.Invoke(
                DeZero.NET.Functions.Mul.Invoke(z, H.Value)[0],
                DeZero.NET.Functions.Mul.Invoke(DeZero.NET.Functions.Sub.Invoke(xp.array(1).ToVariable(), z)[0], h_tilde)[0]
            ).Item1[0];

            //Console.WriteLine($"Output H shape: {string.Join(", ", H.Shape)}");
            return [H.Value];
        }

        public void ResetState()
        {
            H.Value = null;
        }
    }
}
