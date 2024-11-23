using DeZero.NET;
using DeZero.NET.Core;
using DeZero.NET.Extensions;
using System.Runtime.CompilerServices;

namespace MovieFileDataLoaderSampleWorker
{
    public class Conv2dMobileNet : DeZero.NET.Layers.Convolution.Conv2d
    {
        private NDarray _cachedCol;
        private Shape _lastInputShape;
        private readonly int _cacheSize = 5;
        private readonly Dictionary<string, (NDarray col, Shape shape)> _colCache
            = new Dictionary<string, (NDarray col, Shape shape)>();

        public Conv2dMobileNet(int out_channels, int kernel_size, Dtype dtype,
            int stride = 1, int pad = 0, bool nobias = false, int? in_channels = null)
            : base(out_channels, kernel_size, dtype, stride, pad, nobias, in_channels)
        {
            // 初期化時に重みを最適化
            if (W?.Value != null)
            {
                OptimizeWeights();
            }
        }

        private void OptimizeWeights()
        {
            if (W?.Value?.Data?.Value is not null)
            {
                // 重みを最適なメモリレイアウトに再配置
                W.Value.Data.Value = xp.ascontiguousarray(W.Value.Data.Value);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private string GetCacheKey(Shape inputShape)
        {
            return $"{inputShape[0]}_{inputShape[1]}_{inputShape[2]}_{inputShape[3]}";
        }

        public override Variable[] Forward(params Variable[] xs)
        {
            if (!ValidateInputs(xs, out var x)) return null;

            if (NeedsWeightInitialization(x))
            {
                InitializeWeights(x);
            }

            // キャッシュの使用を試みる
            var cacheKey = GetCacheKey(x.Shape);
            NDarray col;

            if (_colCache.TryGetValue(cacheKey, out var cached) &&
                cached.shape.Dimensions.SequenceEqual(x.Shape.Dimensions))
            {
                col = cached.col.copy();
            }
            else
            {
                // 新しいcolを計算
                using var scope = new ComputationScope();
                col = Utils.im2col_array(x, (W.Value.Shape[2], W.Value.Shape[3]),
                    (Stride.Value, Stride.Value), (Pad.Value, Pad.Value), to_matrix: false).Data.Value;

                // キャッシュの管理
                ManageCache(cacheKey, col, x.Shape);
            }

            // 最適化された行列演算
            using var computeScope = new ComputationScope();
            var y = ComputeOutput(col, x);
            computeScope.Register(col.ToVariable());

            return new[] { y };
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool ValidateInputs(Variable[] xs, out Variable x)
        {
            x = null;
            if (xs is null || xs.Length == 0 || xs[0]?.Data?.Value is null)
                return false;

            x = xs[0];
            return true;
        }

        private bool NeedsWeightInitialization(Variable x)
        {
            return InChannels == null ||
                   x.Shape[1] != InChannels.Value ||
                   W?.Value?.Data?.Value is null;
        }

        private void InitializeWeights(Variable x)
        {
            InChannels.Value = x.Shape[1];
            W.Value.Data.Value = null;
            _init_W();
            OptimizeWeights();
            WInitialized?.Invoke();
        }

        private void ManageCache(string key, NDarray col, Shape shape)
        {
            if (_colCache.Count >= _cacheSize)
            {
                var oldestKey = _colCache.Keys.First();
                _colCache[oldestKey].col?.Dispose();
                _colCache.Remove(oldestKey);
            }
            _colCache[key] = (col.copy(), shape);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private Variable ComputeOutput(NDarray col, Variable x)
        {
            // 最適化された行列乗算
            var y = xp.tensordot(col, W.Value.Data.Value,
                new int[][] { new int[] { 1, 2, 3 }, new int[] { 1, 2, 3 } });

            y = xp.transpose(y, new int[] { 0, 3, 1, 2 });

            // バイアスの適用を最適化
            if (b?.Value?.Data?.Value is not null)
            {
                using var broadcastedBias = xp.reshape(b.Value.Data.Value,
                    new Shape(1, b.Value.Data.Value.shape[0], 1, 1));
                y = xp.add(y, broadcastedBias);
            }

            return y.ToVariable();
        }

        public void Dispose()
        {
            foreach (var cache in _colCache.Values)
            {
                cache.col?.Dispose();
            }
            _colCache.Clear();
            _cachedCol?.Dispose();
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
