namespace DeZero.NET.Layers.Activation
{
    /// <summary>
    /// シグモイド（Sigmoid function）関数
    /// 活性化関数の一つ
    /// ※この関数を何度も使うと勾配消失問題が発生する
    /// </summary>
    public class Sigmoid : Layer
    {
        public override Variable[] Forward(params Variable[] xs)
        {
            var x = xs[0];
            return Functions.Sigmoid.Invoke(x);
        }
    }
}
