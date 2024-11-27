using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class Round : Function
    {
        public Round()
        {
        }

        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>(0);
            return [x.Data.Value.round().ToVariable(this)];
        }

        public override Variable[] Backward(Params args)
        {
            return [new NDarray(0).ToVariable(this)];
        }

        public static Variable[] Invoke(Variable x)
        {
            return new Round().Call(Params.New.SetPositionalArgs(x));
        }
    }
}
