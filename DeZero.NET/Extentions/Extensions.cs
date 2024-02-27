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

        public static Shape ToShape(this NDarray array, bool useCupy = true)
        {
            switch (useCupy)
            {
                case true:
                    try
                    {
                        array.Push(ArrayMode.cp);
                        return new Shape(array.CupyNDarray.shape);
                    }
                    finally
                    {
                        array.Pop();
                    }
                case false:
                    try
                    {
                        array.Push(ArrayMode.np);
                        return new Shape(array.NumpyNDarray.shape);
                    }
                    finally
                    {
                        array.Pop();
                    }
            }
        }
    }
}
