namespace DeZero.NET
{
    public class Parameter : Variable
    {
        public Parameter(Variable v) : base(v.Data.Value)
        {
        }

        public Parameter(Variable v, string name) : base(v?.Data.Value, name)
        {
        }

        public Variable AsVariable(object obj)
        {
            if (obj is Variable v)
            {
                return v;
            }

            return new Variable((NDarray)obj);
        }
    }
}
