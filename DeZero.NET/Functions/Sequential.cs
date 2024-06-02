using DeZero.NET.Core;

namespace DeZero.NET.Functions
{
    public class Sequential : Function
    {
        private List<Function> _funcs;

        public Sequential(params Function[] funcs)
        {
            _funcs = new List<Function>(funcs);
        }

        public override Variable[] Forward(Params args)
        {
            Variable x = args.Get<Variable>(0);
            foreach (Function func in _funcs)
            {
                x = func.Forward(Params.New.SetPositionalArgs(x))[0];
            }
            return [x];
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Get<Variable>(0);
            for (int i = _funcs.Count - 1; i >= 0; i--)
            {
                gy = _funcs[i].Backward(Params.New.SetPositionalArgs(gy))[0];
            }
            return [gy];
        }
    }
}
