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
    }
}
