using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeZero.NET.Layers
{
    public class Projection : Layer
    {
        public override Func<Variable[], Variable[]> F { get; }

        public Projection(Func<Variable[], Variable[]> f)
        {
            F = f;
        }

        public override Variable[] Forward(params Variable[] xs)
        {
            return F(xs);
        }
    }
}
