using DeZero.NET.Core;

namespace DeZero.NET
{
    public class Function
    {
        private readonly Func<Params, Variable[]> _f;
        public int Generation { get; set; }
        public IEnumerable<Variable> Inputs { get; private set; }
        public IEnumerable<Variable> Outputs { get; private set; }

        protected Function()
        {
        }

        public Function(Func<Params, Variable[]> f)
        {
            _f = f;
        }

        public Variable[] BaseForward(Params args)
        {
            var ys = _f is not null ? _f(args) : Forward(args);

            var outputs = ys.Select(y => (xp.isscalar(y.Data) ? xp.array(y.Data).ToVariable() : y)).ToList();

            if (Config.EnableBackprop)
            {
                Generation = args.Through().Select(x => x.Generation).Max();
                foreach (var output in outputs)
                {
                    output.Creator ??= this;
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
            return args.Through();
        }

        public override int GetHashCode()
        {
            return base.GetHashCode() ^ Generation.GetHashCode();
        }
    }
}
