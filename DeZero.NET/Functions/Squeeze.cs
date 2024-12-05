using DeZero.NET.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class Squeeze : Function
    {
        private readonly int? axis;

        public Squeeze(int? axis = null)
        {
            this.axis = axis;
        }

        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>(0);
            var shape = x.Shape.Dimensions;

            if (axis.HasValue)
            {
                if (shape[axis.Value] != 1)
                {
                    throw new ArgumentException($"Cannot squeeze axis {axis.Value} with size {shape[axis.Value]}");
                }
                shape = shape.Where((s, i) => i != axis.Value).ToArray();
            }
            else
            {
                shape = shape.Where(s => s != 1).ToArray();
            }

            var y = Reshape.Invoke(x, new Shape(shape))[0];
            return [y.Relay(this)];
        }

        public override Variable[] Backward(Params args)
        {
            var gx = args.Get<Variable>(0);
            using var input_0_shape = Inputs.ElementAt(0).NDarray.shape;
            return gx.reshape(input_0_shape);
        }

        public static Variable[] Invoke(Variable x, int? axis = null)
        {
            return new Squeeze(axis).Call(Params.New.SetPositionalArgs(x));
        }
    }
}
