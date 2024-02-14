using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeZero.NET
{
    public class Parameter : Variable
    {
        public Parameter(Variable v) : base(v.Data)
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
