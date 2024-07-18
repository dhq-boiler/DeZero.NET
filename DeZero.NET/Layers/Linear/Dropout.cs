using DeZero.NET.Core;

namespace DeZero.NET.Layers.Linear
{
    public class Dropout : Layer
    {
        public Property<double> DropoutRatio { get; } = new(nameof(DropoutRatio));

        public Dropout(double dropoutRatio)
        {
            RegisterEvent(DropoutRatio);
            DropoutRatio.Value = dropoutRatio;
        }

        public override Variable[] Forward(params Variable[] xs)
        {
            var x = xs[0];
            return [Functions.Dropout.Invoke(x, DropoutRatio.Value)];
        }
    }
}
