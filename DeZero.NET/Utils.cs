namespace DeZero.NET
{
    internal static class Utils
    {
        public static Variable as_variable(object obj)
        {
            if (obj is Variable v)
            {
                return v;
            }

            return new Variable((NDarray)obj);
        }

        public static NDarray sum_to(NDarray x, Shape shape)
        {
            int ndim = shape.Dimentions.Length;
            int lead = x.ndim - ndim;
            int[] leadAxis = Enumerable.Range(0, lead).ToArray();

            int[] axis = Enumerable.Range(0, shape.Dimentions.Length)
                .Where(i => shape[i] == 1)
                .Select(i => i + lead)
                .ToArray();

            // leadAxis と axis を結合
            int[] combinedAxes = leadAxis.Concat(axis).ToArray();

            NDarray y = x.sum(new Axis(combinedAxes), keepdims: true);
            if (lead > 0)
            {
                y = y.squeeze(new Axis(leadAxis));
            }
            return y;
        }
    }
}
