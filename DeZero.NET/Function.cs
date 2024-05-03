using Cupy;
using DeZero.NET.Core;
using Numpy;

namespace DeZero.NET
{
    public class Function
    {
        protected Func<Params, Variable[]> _f;
        public int Generation { get; set; }
        public IEnumerable<Core.Parameter> Inputs { get; private set; }
        public IEnumerable<Variable> Outputs { get; private set; }

        protected Function()
        {
        }

        public Function(Func<Params, Variable[]> f)
        {
            _f = f;
        }

        public virtual Variable[] Call(Params args)
        {
            //var ys = _f is not null ? _f(args) : Forward(args);

            //var inputs = args.Through().Select(x => cpExtensions.as_variable(x)).ToArray();
            //var inputsData = inputs.Select(x => x.Data);
            //var firstShape = inputsData.First().shape;
            //var xs = inputs.Select(x => x.Data);
            //var xs = xp.stack(inputsData.Where(x => firstShape == x.shape).ToArray()).ToVariable();
            //var xs = xp.concatenate(args.Through().Select(x => x.Data).ToArray());
            //args.Set("x", xs);

            //var ys = Forward(Params<Variable>.args(xs, "x").SetParams<Params>(args));
            var ys = Forward(args);

            var outputs = ys.Select(y => (xp.isscalar(y.Data) ? xp.array(y.Data).ToVariable() : y)).ToList();

            if (Config.EnableBackprop)
            {
                Generation = args.Through().Select(x => x.Variable.Generation).Max();
                foreach (var output in outputs)
                {
                    output.Creator = this;
                    this.Inputs = args.Through();
                    this.Outputs = outputs;
                }
            }

            return outputs.ToArray();
        }

        public virtual Variable[] Forward(Params args)
        {
            return _f(args);
        }

        public virtual Variable[] Backward(Params args)
        {
            return args.Through().Select(x => x.Variable).ToArray();
        }

        public override int GetHashCode()
        {
            return base.GetHashCode() ^ Generation.GetHashCode();
        }

        public virtual void ResetParams()
        {
        }
    }
}
