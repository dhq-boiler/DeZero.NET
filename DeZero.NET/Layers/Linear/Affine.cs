namespace DeZero.NET.Layers.Linear
{
    public class Affine : Layer
    {
        public int InputSize { get; set; }
        public int OutputSize { get; set; }

        public Affine(int input_size, int output_size)
        {
            InputSize = input_size;
            OutputSize = output_size;
        }

        public override Variable[] Forward(params Variable[] xs)
        {
            return [Functions.Affine.Invoke(xs[0], InputSize, OutputSize)];
        }
    }
}
