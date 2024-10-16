using DeZero.NET;
using DeZero.NET.Core;

namespace MovieFileDataLoaderSampleWorker
{
    public class Conv2dMobileNet : DeZero.NET.Layers.Convolution.Conv2d
    {
        public Conv2dMobileNet(int out_channels, int kernel_size, Dtype dtype, int stride = 1, int pad = 0, bool nobias = false, int? in_channels = null) : base(out_channels, kernel_size, dtype, stride, pad, nobias, in_channels) { }

        public override Variable[] Forward(params Variable[] xs)
        {
            var x = xs[0];
            // 入力のチャンネル数が InChannels と一致しているかチェック
            if (x.Shape[1] != InChannels.Value)
            {
                InChannels.Value = x.Shape[1];
                //throw new ArgumentException($"Input channel does not match. Expected: {InChannels.Value}, Actual: {x.Shape[1]}");
            }
            if (W.Value.Data.Value is null)
            {
                _init_W(); // 重みが初期化されていない場合は初期化
            }
            WInitialized?.Invoke();

            var y = Conv2dMobileNetFunction.Invoke(xs[0], W.Value, b.Value, stride: Stride.Value, pad: Pad.Value);
            return y;
        }
    }

    public class Conv2dMobileNetFunction : DeZero.NET.Functions.Conv2d
    {
        public Conv2dMobileNetFunction(int stride, int pad) : base(stride, pad)
        {
        }

        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>("x");
            var W = args.Get<Variable>("W");
            var b = args.Get<Variable>("b");
            Shape KH = W.Shape[2], KW = W.Shape[3];
            var col = Utils.im2col_array(x, (KH[0], KW[0]), Stride, Pad, to_matrix: false);
            //Console.WriteLine($"col shape: {string.Join(", ", col.Data.Value.shape)}");
            //Console.WriteLine($"W shape: {string.Join(", ", W.Data.Value.shape)}");

            var y = xp.tensordot(col.Data.Value, W.Data.Value, new int[][] { new int[] { 1, 2, 3 }, new int[] { 1, 2, 3 } });
            //Console.WriteLine($"y shape after tensordot: {string.Join(", ", y.shape)}");

            // 軸の順序を修正
            y = xp.transpose(y, new int[] { 0, 3, 1, 2 });

            if (b is not null)
            {
                //Console.WriteLine($"Bias shape: {string.Join(", ", b.Data.Value.shape)}");
                var broadcastedBias = xp.reshape(b.Data.Value, new Shape(1, b.Data.Value.shape[0], 1, 1));
                //Console.WriteLine($"Broadcasted bias shape: {string.Join(", ", broadcastedBias.shape)}");
                y = xp.add(y, broadcastedBias);
            }

            //Console.WriteLine($"Final y shape: {string.Join(", ", y.shape)}");
            return [y.ToVariable(this)];
        }

        public static Variable[] Invoke(Variable x, Variable W, Variable b = null, int stride = 1, int pad = 0)
        {
            return new Conv2dMobileNetFunction(stride, pad).Call(Params.New.SetKeywordArg(x, W, b));
        }
    }
}
