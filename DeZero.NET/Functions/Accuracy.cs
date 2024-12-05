using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class Accuracy : Function
    {
        public override Variable[] Forward(Params args)
        {
            var y = args.Get<Variable>(0);
            var t = args.Get<Variable>(1);

            using var argmax = y.Data.Value.argmax(axis: 1).ToVariable(y);
            using var pred = argmax.Data.Value.reshape(t.Shape).ToVariable(argmax);
            using var result = (pred.Data.Value == t.Data.Value).ToVariable(pred);
            var acc = result.Data.Value.mean().ToVariable(result);
            return [xp.asarray(acc.Data.Value).ToVariable(acc, this)];
        }

        public static Variable[] Invoke(Variable y, Variable t)
        {
            return new Accuracy().Call(Params.New.SetPositionalArgs(y, t));
        }
    }
}
