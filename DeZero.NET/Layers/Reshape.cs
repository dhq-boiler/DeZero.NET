using DeZero.NET.Core;

namespace DeZero.NET.Layers
{
    public class Reshape : Layer
    {
        public Property<Shape> Shape { get; } = new(nameof(Shape));

        public Reshape(Shape shape)
        {
            RegisterEvent(Shape);
            this.Shape.Value = shape;
        }

        public override Variable[] Forward(params Variable[] xs)
        {
            var x = xs[0];
            return Functions.Reshape.Invoke(x, this.Shape.Value);
        }
    }
}
