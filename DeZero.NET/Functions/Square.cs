using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    /// <summary>
    /// 入力値を2乗する関数を実装します。
    /// </summary>
    public class Square : Function
    {
        /// <summary>
        /// 順伝播の計算を実行します。
        /// </summary>
        /// <param name="inputs">入力値の配列</param>
        /// <returns>2乗された値を含むNDarray配列</returns>
        public override Variable[] Forward(Params inputs)
        {
            var x = inputs.Get<Variable>(0);
            return [(x.Data.Value * x.Data.Value).Relay(this)];
        }

        /// <summary>
        /// 逆伝播の計算を実行します。2乗関数の微分は2x。
        /// </summary>
        /// <param name="inputs">入力値の配列</param>
        /// <param name="grad">上流からの勾配</param>
        /// <returns>計算された勾配を含むNDarray配列</returns>
        public override Variable[] Backward(Params inputs)
        {
            // 入力値の取得
            var x = inputs.Get<Variable>(0);

            // 上流からの勾配の取得
            // this.Outputsから最初の出力の勾配を取得
            var grad = this.Outputs.First().Grad?.Value.Data.Value;

            if (grad is null)
            {
                throw new InvalidOperationException("Gradient is null. Ensure backward propagation is properly initialized.");
            }

            // 勾配の計算: 2x * grad
            return [(2.0f * x.Data.Value * grad).ToVariable()];
        }

        public static Variable[] Invoke(Variable x)
        {
            return new Square().Call(Params.New.SetPositionalArgs(x));
        }
    }
}
