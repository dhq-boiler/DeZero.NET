namespace DeZero.NET.Layers
{
    public class Reshape : Layer
    {
        public Shape Shape { get; private set; }

        public Reshape(Shape shape)
        {
            this.Shape = shape;
        }

        public override Variable[] Forward(params Variable[] xs)
        {
            var x = xs[0];
            return Functions.Reshape.Invoke(x, this.Shape);
        }
    }
}
