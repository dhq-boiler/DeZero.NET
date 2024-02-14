namespace DeZero.NET
{
    public abstract class Function
    {
        public int Generation { get; set; }
        public IEnumerable<Variable> Inputs { get; private set; }
        public IEnumerable<Variable> Outputs { get; private set; }

        public Variable[] Invoke(params Variable[] inputs)
        {
            var _inputs = inputs.Select(x => x).ToList();

            var xs = _inputs;
            var ys = Forward(xs.ToArray());

            var outputs = ys.Select(y => new Variable(xp.isscalar(y) ? xp.array(y.Data) : y.Data));

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

            return [..outputs];
        }

        public abstract Variable[] Forward(params Variable[] xs);
        public abstract Variable[] Backward(params Variable[] gys);

        public override int GetHashCode()
        {
            return base.GetHashCode() ^ Generation.GetHashCode();
        }
    }
}
