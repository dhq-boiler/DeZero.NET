namespace DeZero.NET.Functions
{
    public class Neg : Function
    {

        public override Variable[] Forward(params Variable[] xs)
        {
            return xs.Select(x => -x).ToArray();
        }

        public override Variable[] Backward(params Variable[] gys)
        {
            return gys.Select(gy => -gy).ToArray();
        }

        public static Variable[] Invoke(Variable x)
        {
            return new Neg().Forward(x);
        }
    }
}
