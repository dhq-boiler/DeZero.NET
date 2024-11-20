using DeZero.NET.Core;
using DeZero.NET.Extensions;
using DeZero.NET.Log;
using System.Collections.Concurrent;

namespace DeZero.NET.Layers.Recurrent
{
    public sealed class HighPerformanceGRU : Layer, IDisposable
    {
        private const int PREFETCH_BUFFER_SIZE = 2;
        private const float COMPRESSION_RATIO = 0.5f;
        private readonly int _maxBatchSize;
        private readonly ConcurrentQueue<Variable> _weightCache;
        private readonly object _stateLock = new();
        private volatile bool _isDisposed;
        private readonly ILogger _logger;

        // 主要なGRUのプロパティ
        public Property<Linear.Linear> Wxz { get; }
        public Property<Linear.Linear> Whz { get; }
        public Property<Variable> H { get; }

        public HighPerformanceGRU(int inSize, int hiddenSize, int maxBatchSize = 32, LogLevel minimumLogLevel = LogLevel.Info)
        {
            _logger = new ConsoleLogger(minimumLogLevel, false);
            _maxBatchSize = maxBatchSize;
            _weightCache = new ConcurrentQueue<Variable>();

            // 重みの初期化を最適化
            Wxz = new Property<Linear.Linear>(nameof(Wxz));
            Whz = new Property<Linear.Linear>(nameof(Whz));
            H = new Property<Variable>(nameof(H));

            InitializeWeights(inSize, hiddenSize);
            PrecomputeWeights();
        }

        private void InitializeWeights(int inSize, int hiddenSize)
        {
            _logger.LogDebug($"Initializing GRU weights with inSize={inSize}, hiddenSize={hiddenSize}");

            // 入力変換の重み行列 (inSize → hiddenSize)
            Wxz.Value = new Linear.Linear(out_size: hiddenSize, in_size: inSize);
            SetAttribute("Wxz", Wxz.Value);

            // 隠れ状態の重み行列 (hiddenSize → hiddenSize)
            Whz.Value = new Linear.Linear(out_size: hiddenSize, in_size: hiddenSize, nobias: true);
            SetAttribute("Whz", Whz.Value);

            _logger.LogDebug($"Weight shapes - Wxz: {string.Join(",", Wxz.Value.W.Value.Shape.Dimensions)}, " +
                            $"Whz: {string.Join(",", Whz.Value.W.Value.Shape.Dimensions)}");
        }

        private void InitializeOptimizedWeights(Linear.Linear layer)
        {
            // Xavier初期化を使用
            var scale = MathF.Sqrt((2.0f / (layer.InSize.Value + layer.OutSize.Value)).Value);
            layer.W.Value = new Parameter(new Variable(xp.random.normal(new NDarray<float>(0f), new NDarray<float>(scale), [layer.OutSize.Value, layer.InSize.Value.Value])));
            if (!layer.NoBias)
            {
                layer.b.Value = new Parameter(xp.zeros(new Shape(layer.OutSize.Value)).ToVariable());
            }
        }

        private void PrecomputeWeights()
        {
            // 事前計算された重みをキャッシュ
            for (int i = 0; i < PREFETCH_BUFFER_SIZE; i++)
            {
                var precomputed = ComputeCombinedWeights();
                _weightCache.Enqueue(precomputed);
            }
        }

        private Variable ComputeCombinedWeights()
        {
            try
            {
                var wxz = Wxz.Value.W.Value.Data.Value;
                var whz = Whz.Value.W.Value.Data.Value;

                _logger.LogDebug($"Original Wxz shape: {string.Join(",", wxz.shape)}");
                _logger.LogDebug($"Original Whz shape: {string.Join(",", whz.shape)}");

                // 必要に応じて転置
                // wxz: (4, 32) -> (32, 4)
                var wxz_t = wxz.transpose();
                // whz: (32, 32) はそのまま

                _logger.LogDebug($"Wxz after transpose: {string.Join(",", wxz_t.shape)}");
                _logger.LogDebug($"Whz shape: {string.Join(",", whz.shape)}");

                // 横方向（axis=1）に結合
                // wxz_t: (32, 4), whz: (32, 32) -> result: (32, 36)
                var combined = xp.concatenate(new[] { wxz_t, whz }, axis: 1);
                _logger.LogDebug($"Combined weights shape: {string.Join(",", combined.shape)}");

                return combined.copy().ToVariable();
            }
            catch (Exception ex)
            {
                _logger.LogError($"Failed to combine weights: {ex.Message}");
                throw;
            }
        }

        public override Variable[] Forward(params Variable[] inputs)
        {
            ThrowIfDisposed();
            var x = inputs[0];

            if (x.Shape[0] > _maxBatchSize)
            {
                return ProcessLargeBatch(x);
            }

            return ProcessRegularBatch(x);
        }

        private Variable[] ProcessRegularBatch(Variable x)
        {
            //using var scope = new ComputationScope();

            //// 入力の次元を確認・修正
            //x = DimensionHelper.EnsureShape(x, 2, _logger);

            //// キャッシュされた重みを使用
            //if (!_weightCache.TryDequeue(out var weights))
            //{
            //    weights = ComputeCombinedWeights();
            //}
            //scope.Register(weights);

            ////// 行列の次元を検証
            ////DimensionHelper.ValidateMatrixDimensions(x, weights, "GRU Forward", _logger);

            //// 最適化されたGRU計算
            //var (newState, isValid) = ComputeGRUStateOptimized(x, weights);
            //if (!isValid)
            //{
            //    return new[] { CreateZeroState(x.Shape[0]) };
            //}

            //// 新しい重みを事前計算してキャッシュに追加
            //Task.Run(() =>
            //{
            //    var newWeights = ComputeCombinedWeights();
            //    _weightCache.Enqueue(newWeights);
            //});

            //return new[] { newState };

            using var scope = new ComputationScope();

            x = DimensionHelper.EnsureShape(x, 2, _logger);

            if (!_weightCache.TryDequeue(out var weights))
            {
                // 同期的に重みを計算
                weights = ComputeCombinedWeights();

                //// バックグラウンドで次の重みを準備（別の重みインスタンスを作成）
                //_ = Task.Run(() =>
                //{
                    try
                    {
                        var nextWeights = ComputeCombinedWeights();
                        _weightCache.Enqueue(nextWeights);
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError($"Failed to precompute weights: {ex.Message}");
                    }
                //}).ConfigureAwait(false);
            }
            scope.Register(weights);

            var (newState, isValid) = ComputeGRUStateOptimized(x, weights);
            if (!isValid)
            {
                return new[] { CreateZeroState(x.Shape[0]) };
            }

            return new[] { newState };
        }

        private (Variable state, bool isValid) ComputeGRUStateOptimized(Variable x, Variable weights)
        {
            try
            {
                using var scope = new ComputationScope();

                _logger.LogDebug($"Input x shape: {string.Join(",", x.Shape.Dimensions)}");
                _logger.LogDebug($"Weights shape: {string.Join(",", weights.Shape.Dimensions)}");

                // x: (32, 4) を転置して (4, 32)にする
                var x_t = Functions.Transpose.Invoke(x)[0];
                scope.Register(x_t);
                _logger.LogDebug($"Transposed x shape: {string.Join(",", x_t.Shape.Dimensions)}");

                // 行列積の計算
                // x_t: (4, 32), weights: (32, 36) -> result: (4, 36)
                var combined = DeZero.NET.Functions.MatMul.Invoke(x_t, weights)[0];
                scope.Register(combined);

                // 結果を転置して(36, 4)にする
                var combined_t = Functions.Transpose.Invoke(combined)[0];
                scope.Register(combined_t);
                _logger.LogDebug($"Combined shape after transpose: {string.Join(",", combined_t.Shape.Dimensions)}");

                // Activation関数を適用
                var activated = DeZero.NET.Functions.Sigmoid.Invoke(combined_t)[0];

                _logger.LogDebug($"Output shape: {string.Join(",", activated.Shape.Dimensions)}");

                return (activated, true);
            }
            catch (Exception ex)
            {
                _logger.LogError($"GRU computation error: {ex.Message}");
                return (null, false);
            }
        }

        private Variable[] ProcessLargeBatch(Variable x)
        {
            var results = new List<Variable>();
            var batchSize = x.Shape[0];

            // バッチを並列処理
            Parallel.For(0, (batchSize + _maxBatchSize - 1) / _maxBatchSize, i =>
            {
                var start = i * _maxBatchSize;
                var end = Math.Min(start + _maxBatchSize, batchSize);
                var batch = x.Data.Value.Slice(new[] { new Slice(start, end) }).ToVariable();
                var result = ProcessRegularBatch(batch)[0];
                lock (results)
                {
                    results.Add(result);
                }
            });

            // 結果を結合
            return new[] { CombineResults(results) };
        }

        private Variable CreateZeroState(int batchSize)
        {
            return new Variable(xp.zeros(new Shape(batchSize, Whz.Value.OutSize.Value)));
        }

        private Variable CombineResults(List<Variable> results)
        {
            return DeZero.NET.Functions.Concatenate.Invoke(results.ToArray(), axis: 0)[0];
        }

        private void ThrowIfDisposed()
        {
            if (_isDisposed)
            {
                throw new ObjectDisposedException(nameof(HighPerformanceGRU));
            }
        }

        public void Dispose()
        {
            if (_isDisposed) return;
            _isDisposed = true;

            while (_weightCache.TryDequeue(out var weights))
            {
                weights?.Dispose();
            }

            H.Value?.Dispose();
            GC.SuppressFinalize(this);
        }

        public void ResetState()
        {
            lock (_stateLock)
            {
                H.Value?.Dispose();
                H.Value = null;
            }
        }

        /// <summary>
        /// キャッシュをクリアするためのオプション
        /// </summary>
        [Flags]
        public enum CacheClearOptions
        {
            /// <summary>
            /// 重みのキャッシュのみをクリア
            /// </summary>
            WeightsOnly = 1,

            /// <summary>
            /// 計算用の一時データをクリア
            /// </summary>
            ComputationCache = 2,

            /// <summary>
            /// すべてのキャッシュをクリア
            /// </summary>
            All = WeightsOnly | ComputationCache
        }

        /// <summary>
        /// キャッシュされたデータをクリアし、必要に応じて再初期化を行います。
        /// </summary>
        /// <param name="options">クリアするキャッシュの種類を指定するオプション</param>
        /// <param name="regenerateWeightCache">重みキャッシュを再生成するかどうか</param>
        public void ClearCache(CacheClearOptions options = CacheClearOptions.All, bool regenerateWeightCache = true)
        {
            lock (_stateLock)
            {
                try
                {
                    if ((options & CacheClearOptions.WeightsOnly) != 0)
                    {
                        ClearWeightCache();
                    }

                    if ((options & CacheClearOptions.ComputationCache) != 0)
                    {
                        ClearComputationCache();
                    }

                    if (regenerateWeightCache && (options & CacheClearOptions.WeightsOnly) != 0)
                    {
                        RegenerateWeightCache();
                    }

                    // 明示的にGCを実行
                    GC.Collect(1, GCCollectionMode.Optimized);
                }
                catch (Exception ex)
                {
                    throw new InvalidOperationException("Failed to clear cache", ex);
                }
            }
        }

        private void ClearWeightCache()
        {
            while (_weightCache.TryDequeue(out var weights))
            {
                weights?.Dispose();
            }
        }

        private void ClearComputationCache()
        {
            // 計算用の一時データをクリア
            H.Value?.Dispose();
            H.Value = null;
        }

        private void RegenerateWeightCache()
        {
            for (int i = 0; i < PREFETCH_BUFFER_SIZE; i++)
            {
                var precomputed = ComputeCombinedWeights();
                _weightCache.Enqueue(precomputed);
            }
        }

        /// <summary>
        /// キャッシュの状態を診断し、メモリ使用量などの情報を取得します。
        /// </summary>
        /// <returns>キャッシュの診断情報</returns>
        public CacheDiagnostics GetCacheDiagnostics()
        {
            return new CacheDiagnostics
            {
                WeightCacheCount = _weightCache.Count,
                HasCurrentState = H.Value != null,
                TotalCachedItems = _weightCache.Count + (H.Value != null ? 1 : 0)
            };
        }

        /// <summary>
        /// キャッシュの診断情報を保持するクラス
        /// </summary>
        public class CacheDiagnostics
        {
            public int WeightCacheCount { get; set; }
            public bool HasCurrentState { get; set; }
            public int TotalCachedItems { get; set; }
        }
    }
}
