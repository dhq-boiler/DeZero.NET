namespace DeZero.NET
{
    public static class Extensions
    {
        public static Variable ToVariable(this NDarray array, bool useCupy = true)
        {
            var ret = new Variable(array);
            switch (useCupy)
            {
                case true when ret.Data.CupyNDarray is null:
                case false when ret.Data.NumpyNDarray is null:
                    ret.Data.Switch();
                    break;
            }

            return ret;
        }
    }
}
