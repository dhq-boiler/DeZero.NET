using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Layers.Recurrent
{
    public class GRU : Layer
    {
        private Queue<Variable> _stateHistory = new Queue<Variable>();
        private const int MAX_STATE_HISTORY = 60;
        private Dictionary<string, Variable> _weightCache;
        private float _compressionRate = 0.8f;

        // 既存のプロパティ
        public Property<DeZero.NET.Layers.Linear.Linear> Wxz { get; } = new(nameof(Wxz));
        public Property<DeZero.NET.Layers.Linear.Linear> Wxr { get; } = new(nameof(Wxr));
        public Property<DeZero.NET.Layers.Linear.Linear> Wxh { get; } = new(nameof(Wxh));
        public Property<DeZero.NET.Layers.Linear.Linear> Whz { get; } = new(nameof(Whz));
        public Property<DeZero.NET.Layers.Linear.Linear> Whr { get; } = new(nameof(Whr));
        public Property<DeZero.NET.Layers.Linear.Linear> Whh { get; } = new(nameof(Whh));
        public Property<Variable> H { get; } = new(nameof(H));

        // 新しい最適化関連のプロパティ
        public bool EnableStateCompression { get; set; }
        public bool BatchProcessingEnabled { get; set; }
        public int BatchSize { get; set; }
        public bool EnableWeightCaching { get; set; }
        public int CacheSize { get; set; }
        private readonly object _cacheLock = new object();

        public GRU(int inSize, int hiddenSize)
        {
            RegisterEvent(Wxz, Wxr, Wxh, Whz, Whr, Whh, H);

            // 既存の初期化
            InitializeWeights(inSize, hiddenSize);

            // 新しい初期化
            InitializeOptimizations();

            // 初期化後の実際のサイズを確認
            Console.WriteLine($"GRU Initialized - Wxz in_size: {Wxz.Value.InSize.Value}, out_size: {Wxz.Value.OutSize.Value}");
            Console.WriteLine($"GRU Initialized - Whz in_size: {Whz.Value.InSize.Value}, out_size: {Whz.Value.OutSize.Value}");
        }

        private void InitializeWeights(int inSize, int hiddenSize)
        {
            Wxz.Value = new DeZero.NET.Layers.Linear.Linear(in_size: inSize, out_size: hiddenSize);
            SetAttribute("Wxz", Wxz.Value);
            Wxr.Value = new DeZero.NET.Layers.Linear.Linear(in_size: inSize, out_size: hiddenSize);
            SetAttribute("Wxr", Wxr.Value);
            Wxh.Value = new DeZero.NET.Layers.Linear.Linear(in_size: inSize, out_size: hiddenSize);
            SetAttribute("Wxh", Wxh.Value);
            Whz.Value = new DeZero.NET.Layers.Linear.Linear(in_size: hiddenSize, out_size: hiddenSize, nobias: true);
            SetAttribute("Whz", Whz.Value);
            Whr.Value = new DeZero.NET.Layers.Linear.Linear(in_size: hiddenSize, out_size: hiddenSize, nobias: true);
            SetAttribute("Whr", Whr.Value);
            Whh.Value = new DeZero.NET.Layers.Linear.Linear(in_size: hiddenSize, out_size: hiddenSize, nobias: true);
            SetAttribute("Whh", Whh.Value);
        }

        private void InitializeOptimizations()
        {
            EnableStateCompression = false;
            BatchProcessingEnabled = false;
            BatchSize = 32;
            EnableWeightCaching = false;
            CacheSize = 1000;
            _weightCache = new Dictionary<string, Variable>();
        }

        public override Variable[] Forward(params Variable[] variables)
        {
            try
            {
                var x = variables[0];

                if (BatchProcessingEnabled && x.Shape[0] > BatchSize)
                {
                    return ProcessLargeBatch(x);
                }

                var batchSize = x.Shape[0];

                if (H.Value == null || H.Value.Shape[0] != batchSize)
                {
                    CleanupOldStates();
                    // ここで128次元で初期化されている可能性
                    Console.WriteLine($"Creating new H.Value with size: {Wxz.Value.OutSize.Value}");
                    H.Value = xp.zeros(new Shape(batchSize, Wxz.Value.OutSize.Value), dtype: Dtype.float32).ToVariable();
                }

                var result = CalculateNextState(x);

                if (EnableStateCompression)
                {
                    result = CompressState(result);
                }

                ManageStateHistory(result);

                return [result];
            }
            catch (Exception ex)
            {
                Console.WriteLine($"GRU Forward Error: {ex.Message}");
                throw;
            }
        }

        private Variable CalculateNextState(Variable x)
        {
            using (var scope = new ComputationScope())
            {
                Console.WriteLine($"CalculateNextState input shape: {string.Join(", ", x.Shape.Dimensions)}");

                // 重み計算の結果（Forward の出力）は Dispose 対象
                var wxz_out = Wxz.Value.Forward(x)[0];
                var whz_out = Whz.Value.Forward(H.Value)[0];

                // Forward の出力を Register して管理
                scope.Register(wxz_out);
                scope.Register(whz_out);

                Console.WriteLine($"wxz shape: {string.Join(", ", wxz_out.Shape.Dimensions)}");

                var z = scope.Register(DeZero.NET.Functions.Sigmoid.Invoke(
                    DeZero.NET.Functions.Add.Invoke(wxz_out, whz_out).Item1[0])[0]);

                var wxr_out = Wxr.Value.Forward(x)[0];
                var whr_out = Whr.Value.Forward(H.Value)[0];
                scope.Register(wxr_out);
                scope.Register(whr_out);

                var r = scope.Register(DeZero.NET.Functions.Sigmoid.Invoke(
                    DeZero.NET.Functions.Add.Invoke(wxr_out, whr_out).Item1[0])[0]);

                var wxh_out = Wxh.Value.Forward(x)[0];
                scope.Register(wxh_out);

                var rh = scope.Register(DeZero.NET.Functions.Mul.Invoke(r, H.Value)[0]);

                var whh_out = Whh.Value.Forward(rh)[0];
                scope.Register(whh_out);

                var h_tilde = scope.Register(DeZero.NET.Functions.Tanh.Invoke(
                    DeZero.NET.Functions.Add.Invoke(wxh_out, whh_out).Item1[0])[0]);

                return DeZero.NET.Functions.Add.Invoke(
                    scope.Register(DeZero.NET.Functions.Mul.Invoke(z, H.Value)[0]),
                    scope.Register(DeZero.NET.Functions.Mul.Invoke(
                        DeZero.NET.Functions.Sub.Invoke(xp.array(1).ToVariable(), z)[0],
                        h_tilde)[0])
                ).Item1[0];
            }
        }

        private Variable[] ProcessLargeBatch(Variable x)
        {
            var batchSize = x.Shape[0];
            var results = new List<Variable>();

            for (int i = 0; i < batchSize; i += BatchSize)
            {
                var endIdx = Math.Min(i + BatchSize, batchSize);
                var batchSlice = x.Data.Value.Slice(new[] { new Slice(i, endIdx) }).ToVariable(x);
                var batchResult = Forward(batchSlice);
                results.Add(batchResult[0]);
            }

            return [ConcatenateVariables(results)];
        }

        private Variable CompressState(Variable state)
        {
            try
            {
                using var data = state.Data.Value;
                var shape = state.Shape;

                Console.WriteLine($"Data shape before SVD: {string.Join(", ", data.shape)}");

                // 圧縮率に基づく次元数
                int compressedDim = (int)(shape[1] * _compressionRate);
                Console.WriteLine($"Compressing from {shape[1]} to {compressedDim} dimensions");

                // SVD計算
                NDarray u = null, s = null, vt = null;
                try
                {
                    (u, s, vt) = xp.linalg.svd(data, compute_uv: true);
                    using var compressed = u.take(new NDarray(compressedDim), axis: 1);
                    var result = compressed.ToVariable();
                    Console.WriteLine($"SVD Output - u shape: {string.Join(", ", u.shape)}");
                    Console.WriteLine($"Compressed shape: {string.Join(", ", result.Shape.Dimensions)}");
                    return result;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"SVD operation failed: {ex.Message}");
                    return state;
                }
                finally
                {
                    // SVDの結果をクリーンアップ
                    u?.Dispose();
                    s?.Dispose();
                    vt?.Dispose();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"CompressState failed: {ex.Message}");
                return state;
            }
        }

        private Variable ConcatenateVariables(List<Variable> variables)
        {
            try
            {
                var arrays = variables.Select(v => v.Data.Value).ToArray();
                var concatenated = xp.concatenate(arrays, axis: 0);
                return concatenated.ToVariable();
            }
            finally
            {
                variables.ForEach(v => v.Dispose());
            }
        }

        public void ClearCache()
        {
            lock (_cacheLock)
            {
                foreach (var cachedValue in _weightCache.Values)
                {
                    cachedValue?.Dispose();
                }
                _weightCache.Clear();
            }
        }

        private Variable GetOrCreateCachedWeight(string key, Func<Variable> creator)
        {
            if (!EnableWeightCaching)
            {
                return creator();
            }

            lock (_cacheLock)
            {
                if (_weightCache.TryGetValue(key, out var cached))
                {
                    return cached;
                }

                var value = creator();
                if (_weightCache.Count >= CacheSize)
                {
                    var oldestKey = _weightCache.Keys.First();
                    _weightCache[oldestKey]?.Dispose();
                    _weightCache.Remove(oldestKey);
                }
                _weightCache[key] = value;
                return value;
            }
        }

        public void SetCompressionRate(float rate)
        {
            if (rate < 0.1f || rate > 1.0f)
            {
                throw new ArgumentException("Compression rate must be between 0.1 and 1.0");
            }
            _compressionRate = rate;
        }

        private void ManageStateHistory(Variable newState)
        {
            _stateHistory.Enqueue(newState);
            if (_stateHistory.Count > MAX_STATE_HISTORY)
            {
                var oldState = _stateHistory.Dequeue();
                oldState?.Dispose();
            }
        }

        private void CleanupOldStates()
        {
            while (_stateHistory.Count > 0)
            {
                var state = _stateHistory.Dequeue();
                state?.Dispose();
            }
            H.Value?.Dispose();
            H.Value = null;
        }

        public void ResetState()
        {
            CleanupOldStates();
        }
    }
}
