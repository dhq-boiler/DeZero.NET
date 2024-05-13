namespace DeZero.NET.Functions
{
    public static class Loss
    {
        public static Variable MeanSquaredError_simple(Variable x0, Variable x1)
        {
            var diff = Sub.Invoke(x0, x1)[0];
            var y = Div.Invoke(Sum.Invoke(diff * diff)[0], new Variable(new NDarray(diff.Data.len)));
            return y[0];
        }

        public static Variable SoftmaxCrossEntropy_simple(Variable x, Variable t)
        {
            var N = x.Shape[0];
            var p = Softmax.Invoke(x)[0];
            p = Clip.Invoke(p, 1e-15, 1.0)[0];
            var log_p = Log.Invoke(p)[0];
            var tlog_p = GetItem.Invoke(log_p, [xp.arange(N), t.Data])[0];
            var y = Div.Invoke(Mul.Invoke(new NDarray(-1).ToVariable(), Sum.Invoke(tlog_p)[0])[0], new NDarray(N).ToVariable())[0];
            return y;
        }
    }
}
