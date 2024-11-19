using DeZero.NET.Core;

namespace DeZero.NET.Layers.Linear
{
    public class Concatenate : Layer
    {
        public Property<int> Axis { get; } = new(nameof(Axis));

        public Concatenate(int axis = 1)
        {
            RegisterEvent(Axis);
            Axis.Value = axis;
        }

        public override Variable[] Forward(params Variable[] xs)
        {
            var x = xs[0];
            return Functions.Concatenate.Invoke(x, Axis.Value);
        }
    }
}
