namespace DeZero.NET
{
    public static class Extensions
    {
        public static Variable ToVariable(this NDarray array)
        {
            return new Variable(array);
        }
    }
}
