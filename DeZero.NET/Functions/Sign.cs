using DeZero.NET.Core;

namespace DeZero.NET.Functions
{
    public class Sign : Function
    {
        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>(0);
            var y = x.Data.Value.sign();
            return [y.ToVariable(this)];
        }

        public override Variable[] Backward(Params args)
        {
            return [new NDarray(0f).ToVariable(this)];
        }

        public static Variable[] Invoke(Variable x)
        {
            return new Sign().Call(Params.New.SetPositionalArgs(x));
        }
    }
}
