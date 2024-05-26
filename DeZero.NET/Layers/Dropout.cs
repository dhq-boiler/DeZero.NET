namespace DeZero.NET.Layers
{
    public class Dropout : Layer
    {
        public double DropoutRatio { get; set; }

        public Dropout(double dropoutRatio)
        {
            this.DropoutRatio = dropoutRatio;
        }

        public override Variable[] Forward(params Variable[] xs)
        {
            var x = xs[0];
            return [Functions.Dropout.Invoke(x, DropoutRatio)];
        }
    }
}
