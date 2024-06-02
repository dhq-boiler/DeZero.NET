namespace DeZero.NET.Layers.Activation
{
    /// <summary>
    /// Leaky ReLU（Leaky Rectified Linear Unit）関数
    /// 活性化関数の一つ
    /// </summary>
    public class LeakyReLU : Layer
    {
        public override Variable[] Forward(params Variable[] xs)
        {
            var x = xs[0];
            return Functions.LeakyReLU.Invoke(x);
        }
    }
}
