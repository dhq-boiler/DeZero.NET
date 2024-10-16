using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class SliceFunc : Function
    {
        private DeZero.NET.Slice[] _slices;

        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>("x");
            _slices = args.Get<DeZero.NET.Slice[]>("slices");

            var y = x.Data.Value.Slice(_slices);
            return new[] { y.ToVariable(this) };
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Through[0];
            var x = Inputs.ElementAt(0).Variable;

            var gx = xp.zeros(x.Shape, dtype: x.Dtype).ToVariable();

            // Set the sliced part of gx to gy
            gx.Data.Value.SetSlice(_slices, gy.NDarray);

            return new[] { gx };
        }

        public static Variable[] Invoke(Variable x, DeZero.NET.Slice[] slices)
        {
            return new SliceFunc().Call(Params.New.SetKeywordArg(x).SetKeywordArg(slices));
        }
    }
}
