namespace DeZero.NET.Layers.Activation
{
    /// <summary>
    /// ハイパボリックタンジェント（Hyperbolic tangent function、双曲線正接）関数
    /// 活性化関数の一つ
    /// </summary>
    public class Tanh : Layer
    {
        public override Variable[] Forward(params Variable[] xs)
        {
            var x = xs[0];
            return Functions.Tanh.Invoke(x);
        }
    }
}
