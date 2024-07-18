namespace DeZero.NET.Layers.Activation
{
    /// <summary>
    /// ソフトマックス（Softmax）関数
    /// 活性化関数の一つ
    /// 確率分布を出力したい時に使う
    /// </summary>
    public class Softmax : Layer
    {
        public override Variable[] Forward(params Variable[] xs)
        {
            var x = xs[0];
            return Functions.Softmax.Invoke(x);
        }
    }
}
