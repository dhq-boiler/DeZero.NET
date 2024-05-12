using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DeZero.NET.Core;

namespace DeZero.NET.Functions
{
    public class MeanSquaredError : Function
    {
        public override Variable[] Forward(Params args)
        {
            var x0 = args.Get<Variable>(0);
            var x1 = args.Get<Variable>(1);
            var diff = x0.Data - x1.Data;
            var y = (diff * diff).sum() / diff.len;
            return [y.ToVariable(this)];
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Get<Variable>(0);
            var x0 = Inputs.ElementAt(0).Variable;
            var x1 = Inputs.ElementAt(1).Variable;
            var diff = x0.Data - x1.Data;
            var gx0 = gy * diff * (2f / diff.len);
            var gx1 = -gx0;
            return [gx0, gx1];
        }

        public static Variable[] Invoke(Variable x0, Variable x1)
        {
            return new MeanSquaredError().Call(Params.New.SetPositionalArgs(x0, x1));
        }
    }
}
