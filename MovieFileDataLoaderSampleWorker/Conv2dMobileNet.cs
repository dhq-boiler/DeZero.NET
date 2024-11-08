using DeZero.NET;
using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace MovieFileDataLoaderSampleWorker
{
    public class Conv2dMobileNet : DeZero.NET.Layers.Convolution.Conv2d
    {
        public Conv2dMobileNet(int out_channels, int kernel_size, Dtype dtype, int stride = 1, int pad = 0, bool nobias = false, int? in_channels = null) : base(out_channels, kernel_size, dtype, stride, pad, nobias, in_channels) { }

        public override Variable[] Forward(params Variable[] xs)
        {
            if (xs == null || xs.Length == 0)
                throw new ArgumentException("Input array cannot be null or empty");

            var x = xs[0];
            if (x == null)
                throw new ArgumentException("Input variable cannot be null");

            if (x.Shape == null || x.Shape.Dimensions.Length < 2)
                throw new ArgumentException("Input shape is invalid");

            if (InChannels == null)
                throw new InvalidOperationException("InChannels is not initialized");

            if (x.Shape[1] != InChannels.Value)
            {
                InChannels.Value = x.Shape[1];
                W.Value.Data.Value = null;
                _init_W();
            }

            if (W?.Value?.Data?.Value is null)
            {
                _init_W();
            }

            WInitialized?.Invoke();

            return Conv2dMobileNetFunction.Invoke(x, W.Value, b?.Value, stride: Stride.Value, pad: Pad.Value);
        }
    }

    public class Conv2dMobileNetFunction : DeZero.NET.Functions.Conv2d
    {
        public Conv2dMobileNetFunction(int stride, int pad) : base(stride, pad)
        {
        }

        public override Variable[] Forward(Params args)
        {
            if (args == null)
                throw new ArgumentNullException(nameof(args));

            var x = args.Get<Variable>("x");
            var W = args.Get<Variable>("W");
            var b = args.Get<Variable>("b");

            if (x?.Data?.Value is null)
                throw new ArgumentException("Input variable x is null or has null data");
            if (W?.Data?.Value is null)
                throw new ArgumentException("Weight variable W is null or has null data");
            if (W.Shape == null || W.Shape.Dimensions.Length < 4)
                throw new ArgumentException("Weight shape is invalid");

            int kernelHeight = W.Shape[2];
            int kernelWidth = W.Shape[3];

            if (kernelHeight <= 0 || kernelWidth <= 0)
                throw new ArgumentException("Invalid kernel dimensions");

            var col = Utils.im2col_array(x, (kernelHeight, kernelWidth), Stride, Pad, to_matrix: false);
            if (col?.Data?.Value is null)
                throw new InvalidOperationException("im2col_array returned null result");

            var y = xp.tensordot(col.Data.Value, W.Data.Value,
                new int[][] { new int[] { 1, 2, 3 }, new int[] { 1, 2, 3 } });

            y = xp.transpose(y, new int[] { 0, 3, 1, 2 });

            if (b?.Data?.Value is not null)
            {
                var broadcastedBias = xp.reshape(b.Data.Value, new Shape(1, b.Data.Value.shape[0], 1, 1));
                y = xp.add(y, broadcastedBias);
            }

            return new[] { y.ToVariable(this) };
        }

        public static Variable[] Invoke(Variable x, Variable W, Variable b = null, int stride = 1, int pad = 0)
        {
            return new Conv2dMobileNetFunction(stride, pad).Call(Params.New.SetKeywordArg(x, W, b));
        }
    }
}
