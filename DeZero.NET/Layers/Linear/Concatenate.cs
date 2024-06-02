namespace DeZero.NET.Layers.Linear
{
    public class Concatenate : Layer
    {
        public int Axis { get; }

        public Concatenate(int axis = 1)
        {
            Axis = axis;
        }

        public override Variable[] Forward(params Variable[] xs)
        {
            var x = xs[0];
            return [Functions.Concatenate.Invoke(x, Axis)];
        }
    }
}
