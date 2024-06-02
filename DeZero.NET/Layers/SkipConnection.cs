namespace DeZero.NET.Layers
{
    public class SkipConnection : Layer
    {
        public Function Func { get; }

        public SkipConnection(Function func)
        {
            Func = func;
        }

        public override Variable[] Forward(params Variable[] xs)
        {
            return new Functions.SkipConnection(Func).Forward(Core.Params.New.SetPositionalArgs(xs[0]));
        }
    }
}
