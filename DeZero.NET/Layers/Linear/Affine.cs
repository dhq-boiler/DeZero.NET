using DeZero.NET.Core;

namespace DeZero.NET.Layers.Linear
{
    public class Affine : Layer
    {
        public Property<int> InputSize { get; } = new(nameof(InputSize));
        public Property<int> OutputSize { get; } = new(nameof(OutputSize));

        public Affine(int input_size, int output_size)
        {
            RegisterEvent(InputSize, OutputSize);
            InputSize.Value = input_size;
            OutputSize.Value = output_size;
        }

        public override Variable[] Forward(params Variable[] xs)
        {
            return [Functions.Affine.Invoke(xs[0], InputSize.Value, OutputSize.Value)];
        }
    }
}
