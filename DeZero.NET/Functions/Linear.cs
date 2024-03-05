namespace DeZero.NET.Functions
{
    public class Linear : Function
    {
        public override Variable[] Forward(params Variable[] xs)
        {
            var x = xs[0];
            var W = xs[1];
            var b = xs[2];
            var y = x.Data.dot(W.Data);
            if (b is not null)
            {
                y += b.Data;
            }

            return [y.ToVariable()];
        }

        public override Variable[] Backward(params Variable[] gys)
        {
            var gy = gys[0];
            var x = Inputs.ElementAt(0);
            var W = Inputs.ElementAt(1);
            var b = Inputs.ElementAt(2);
            var gb = b.Data is null ? null : SumTo.Invoke(gy, b.Shape)[0];
            var gx = MatMul.Invoke(gy, W.T)[0];
            var gW = MatMul.Invoke(x.T, gy)[0];
            return [gx, gW, gb];
        }

        public static Variable[] Invoke(Variable x, Variable W, Variable b)
        {
            return new Linear().BaseForward(x, W, b);
        }
    }
}
