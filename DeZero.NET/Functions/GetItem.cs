using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class GetItem : Function
    {
        public NDarray[] Slices { get; }

        public GetItem(NDarray[] slices)
        {
            Slices = slices;
        }

        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>(0);
            var y = x.Data.Value[Slices].ToVariable();
            return [y];
        }

        public override Variable[] Backward(Params args)
        {
            var x = Inputs.ElementAt(0);
            var f = new GetItemGrad(Slices, x.Variable.Shape);
            return f.Call(Params.New.SetKeywordArg(args.Get<Variable>(0), "gy"));
        }

        public static Variable[] Invoke(Variable x, params NDarray[] slices)
        {
            return new GetItem(slices).Call(Params.New.SetPositionalArgs(x));
        }
    }
}
