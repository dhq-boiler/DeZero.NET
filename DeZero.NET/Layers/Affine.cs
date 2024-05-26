namespace DeZero.NET.Layers
{
    public class Affine : Layer
    {
        public int InputSize { get; set; }
        public int OutputSize { get; set; }

        public Affine(int input_size, int output_size)
        {
            this.InputSize = input_size;
            this.OutputSize = output_size;
        }

        public override Variable[] Forward(params Variable[] xs)
        {
            return [Functions.Affine.Invoke(xs[0], InputSize, OutputSize)];
        }
    }
}
