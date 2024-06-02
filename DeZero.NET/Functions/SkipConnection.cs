using DeZero.NET.Core;

namespace DeZero.NET.Functions
{
    public class SkipConnection : Function
    {
        private Function _func;

        public SkipConnection(Function func)
        {
            _func = func;
        }

        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>(0);
            Variable h = _func.Forward(Params.New.SetPositionalArgs(x))[0];
            return [x + h];
        }

        public override Variable[] Backward(Params args)
        {
            var gout = args.Get<Variable>(0);
            // goutは上流から伝播されてきた勾配
            Variable gx = gout;  // xに対する勾配
            Variable gh = gout;  // hに対する勾配

            // スキップ接続のBackwardでは、上流から伝播されてきた勾配をそのまま下流に流す
            // 同時に、funcのBackwardメソッドを呼び出して、hに対する勾配を計算する
            gh = _func.Backward(Params.New.SetPositionalArgs(gh))[0];

            // xに対する勾配とhに対する勾配を足し合わせて、最終的な勾配を求める
            gx = gx + gh;

            return [gx];
        }
    }
}
