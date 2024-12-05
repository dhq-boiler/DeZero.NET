using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class Round : Function
    {
        private Shape input_shape;  // 入力の形状を保存

        public Round()
        {
        }

        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>(0);
            input_shape = x.Shape;  // 入力の形状を保存
            return [x.Data.Value.round().Relay(this, x)];
        }

        public override Variable[] Backward(Params args)
        {
            // 入力と同じ形状のゼロ配列を作成
            return [xp.zeros(input_shape).ToVariable(this)];
        }

        public static Variable[] Invoke(Variable x)
        {
            return new Round().Call(Params.New.SetPositionalArgs(x));
        }
    }
}
