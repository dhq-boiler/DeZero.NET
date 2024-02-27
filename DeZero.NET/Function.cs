namespace DeZero.NET
{
    public class Function
    {
        private readonly Func<Variable[], Variable[]> _f;
        public int Generation { get; set; }
        public IEnumerable<Variable> Inputs { get; private set; }
        public IEnumerable<Variable> Outputs { get; private set; }

        protected Function()
        {
        }

        public Function(Func<Variable[], Variable[]> f)
        {
            _f = f;
        }

        public Variable[] BaseForward(params Variable[] inputs)
        {
            var _inputs = inputs.Select(x => x).ToList();

            var xs = _inputs;
            var ys = Forward(xs.ToArray());

            var outputs = ys.Select(y => new Variable(xp.isscalar(y.Data) ? xp.array(y.Data) : y.Data)).ToList();

            if (Config.EnableBackprop)
            {
                Generation = inputs.Select(x => x.Generation).Max();
                foreach (var output in outputs)
                {
                    output.Creator = this;
                    this.Inputs = _inputs;
                    this.Outputs = outputs;
                }
            }

            return outputs.ToArray();
        }

        public virtual Variable[] Forward(params Variable[] xs)
        {
            return _f(xs);
        }

        public virtual Variable[] Backward(params Variable[] gys)
        {
            return gys;
        }

        public override int GetHashCode()
        {
            return base.GetHashCode() ^ Generation.GetHashCode();
        }
    }
}
