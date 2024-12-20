﻿using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class Dropout : Function
    {
        public double DropoutRatio { get; set; }
        public NDarray Mask { get; set; }

        public Dropout(double dropoutRatio)
        {
            this.DropoutRatio = dropoutRatio;
        }

        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>(0);
            if (Config.Train)
            {
                using var z = xp.random.rand(x.Shape.Dimensions);
                if (Mask is not null)
                {
                    Mask.Dispose();
                    Mask = null;
                }
                Mask = z > DropoutRatio;
                using var array = xp.array(1.0 - DropoutRatio);
                using var scale = array.astype(x.Dtype);
                var y = x * Mask / scale;
                return [y.Relay(this)];
            }
            else
            {
                return [x.Relay(this)];
            }
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Get<Variable>(0);
            return [gy * Mask];
        }

        public static Variable Invoke(Variable x, double dropoutRatio = 0.5)
        {
            return new Dropout(dropoutRatio).Call(Params.New.SetPositionalArgs(x))[0];
        }
    }
}
