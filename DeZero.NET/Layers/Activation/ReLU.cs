namespace DeZero.NET.Layers.Activation
{
    /// <summary>
    /// ReLU（Rectified Linear Unit）関数
    /// 活性化関数の一つ
    /// </summary>
    public class ReLU : Layer
    {
        public override Variable[] Forward(params Variable[] xs)
        {
            var x = xs[0];
            return Functions.ReLU.Invoke(x);
        }
    }
}
