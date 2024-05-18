using DeZero.NET.Core;

namespace DeZero.NET.Functions
{
    public class Accuracy : Function
    {
        public override Variable[] Forward(Params args)
        {
            var y = args.Get<Variable>("y");
            var t = args.Get<Variable>("t");

            var pred = y.Data.argmax(axis: 1).reshape(t.Shape);
            var result = (pred == t.Data);
            var acc = result.mean();
            return [xp.asarray(acc).ToVariable(this)];
        }

        public static Variable[] Invoke(Variable y, Variable t)
        {
            return new Accuracy().Call(Params.New.SetPositionalArgs(y, t));
        }
    }
}
