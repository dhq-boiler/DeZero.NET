using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using DeZero.NET.Core;
using DeZero.NET.Extensions;
using DeZero.NET.Log;
using DeZero.NET.Layers;

namespace DeZero.NET.Layers.Recurrent
{
    public sealed class OptimizedGRU : Layer, IDisposable
    {
        private const int PREFETCH_BUFFER_SIZE = 3;
        private const float FP16_THRESHOLD = 1e-4f;
        private const int CACHE_CLEANUP_INTERVAL = 100;
        private const int DEFAULT_BATCH_SIZE = 32;

        private readonly struct BatchSegment
        {
            public readonly int StartIdx;
            public readonly int EndIdx;
            public readonly Variable State;

            public BatchSegment(int start, int end, Variable state)
            {
                StartIdx = start;
                EndIdx = end;
                State = state;
            }
        }

        private sealed class WeightCache : IDisposable
        {
            private readonly ConcurrentDictionary<string, Variable> _cache = new();
            private readonly int _maxSize;
            private volatile bool _isDisposed;

            public WeightCache(int maxSize = 1000)
            {
                _maxSize = maxSize;
            }

            public Variable GetOrAdd(string key, Func<Variable> factory)
            {
                if (_isDisposed) throw new ObjectDisposedException(nameof(WeightCache));
                return _cache.GetOrAdd(key, _ => factory());
            }

            public void Clear()
            {
                foreach (var item in _cache)
                {
                    item.Value?.Dispose();
                }
                _cache.Clear();
            }

            public void Dispose()
            {
                if (_isDisposed) return;
                _isDisposed = true;
                Clear();
            }
        }

        // Core GRU properties
        public Property<Linear.Linear> Wxz { get; } = new(nameof(Wxz));
        public Property<Linear.Linear> Wxr { get; } = new(nameof(Wxr));
        public Property<Linear.Linear> Wxh { get; } = new(nameof(Wxh));
        public Property<Linear.Linear> Whz { get; } = new(nameof(Whz));
        public Property<Linear.Linear> Whr { get; } = new(nameof(Whr));
        public Property<Linear.Linear> Whh { get; } = new(nameof(Whh));
        public Property<Variable> H { get; } = new(nameof(H));

        // Optimization properties
        private readonly ConcurrentQueue<Variable> _computeBuffer = new();
        private readonly WeightCache _weightCache;
        private readonly Variable[] _precomputedWeights;
        private readonly ConcurrentDictionary<int, Variable> _stateCache = new();
        private readonly object _stateLock = new();
        private readonly ILogger _logger;
        private int _currentBatchIndex;
        private volatile bool _isDisposed;

        public bool UseFP16 { get; set; } = true;
        public int BatchSize { get; set; } = DEFAULT_BATCH_SIZE;
        public bool EnablePrefetch { get; set; } = true;

        public OptimizedGRU(int inSize, int hiddenSize, LogLevel logLevel = LogLevel.Error)
        {
            _logger = new ConsoleLogger(logLevel, false);
            _weightCache = new WeightCache();
            _precomputedWeights = new Variable[PREFETCH_BUFFER_SIZE];

            InitializeWeights(inSize, hiddenSize);
            InitializeOptimizations();
        }

        private void InitializeWeights(int inSize, int hiddenSize)
        {
            Wxz.Value = new Linear.Linear(in_size: inSize, out_size: hiddenSize);
            Wxr.Value = new Linear.Linear(in_size: inSize, out_size: hiddenSize);
            Wxh.Value = new Linear.Linear(in_size: inSize, out_size: hiddenSize);
            Whz.Value = new Linear.Linear(in_size: hiddenSize, out_size: hiddenSize, nobias: true);
            Whr.Value = new Linear.Linear(in_size: hiddenSize, out_size: hiddenSize, nobias: true);
            Whh.Value = new Linear.Linear(in_size: hiddenSize, out_size: hiddenSize, nobias: true);

            RegisterProperties();
        }

        private void InitializeOptimizations()
        {
            if (!EnablePrefetch) return;

            for (int i = 0; i < PREFETCH_BUFFER_SIZE; i++)
            {
                _precomputedWeights[i] = PrecomputeWeightMatrices();
            }
        }

        private void RegisterProperties()
        {
            SetAttribute("Wxz", Wxz.Value);
            SetAttribute("Wxr", Wxr.Value);
            SetAttribute("Wxh", Wxh.Value);
            SetAttribute("Whz", Whz.Value);
            SetAttribute("Whr", Whr.Value);
            SetAttribute("Whh", Whh.Value);
        }

        public override Variable[] Forward(params Variable[] inputs)
        {
            ThrowIfDisposed();

            var x = inputs[0];
            var batchSize = x.Shape[0];

            //using var scope = new BatchComputeScope();
            Variable result;

            if (batchSize > BatchSize)
            {
                result = ProcessLargeBatch(x);
            }
            else
            {
                result = ProcessRegularBatch(x);
            }

            UpdateState(result);
            return new[] { result };
        }

        private Variable ProcessLargeBatch(Variable x)
        {
            var batchSize = x.Shape[0];
            var numTasks = Math.Min(Environment.ProcessorCount, (batchSize + BatchSize - 1) / BatchSize);
            var tasks = new Task<Variable>[numTasks];
            var batchesPerTask = (batchSize + numTasks - 1) / numTasks;

            for (int i = 0; i < numTasks; i++)
            {
                var taskIndex = i;
                var startIdx = taskIndex * batchesPerTask;
                var endIdx = Math.Min(startIdx + batchesPerTask, batchSize);

                tasks[i] = Task.Run(() => ProcessBatchSegment(x, new BatchSegment(startIdx, endIdx, GetCachedState(startIdx, endIdx))));
            }

            Task.WaitAll(tasks);
            return CombineResults(tasks.Select(t => t.Result).ToArray());
        }

        private Variable ProcessRegularBatch(Variable x)
        {
            //using var scope = new ComputationScope();
            var weights = GetOptimizedWeights();
            var state = GetOrCreateState(x.Shape[0]);

            var ret = ComputeGRUOutput(x, weights, state);
            return ret;
        }

        private Variable ProcessBatchSegment(Variable x, BatchSegment segment)
        {
            using var scope = new ComputationScope();

            var batchX = x.Data.Value.Slice(new[] { new Slice(segment.StartIdx, segment.EndIdx) }).ToVariable(x);
            var weights = GetOptimizedWeights();

            return ComputeGRUOutput(batchX, weights, segment.State);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private Variable ComputeGRUOutput(Variable x, Variable weights, Variable state)
        {
            var gates = BatchedGEMM(x, weights, state);
            using var scope = new ComputationScope();  // メモリを自動管理
            scope.Register(gates);

            var z = DeZero.NET.Functions.Sigmoid.Invoke(gates[0])[0];
            var h = DeZero.NET.Functions.Tanh.Invoke(gates[2])[0];
            scope.Register(z, h);

            return DeZero.NET.Functions.Add.Invoke(
                DeZero.NET.Functions.Mul.Invoke(z, state)[0],
                DeZero.NET.Functions.Mul.Invoke(
                    DeZero.NET.Functions.Sub.Invoke(xp.array(1).ToVariable(), z)[0],
                    h
                )[0]
            ).Item1[0];
        }
        //private Variable ComputeGRUOutput(Variable x, Variable weights, Variable state)
        //{
        //    using var scope = new ComputationScope();

        //    // Gates computation
        //    var gates = UseFP16 ? ComputeGatesFP16(x, weights, state) : ComputeGatesFP32(x, weights, state);

        //    // Update gate
        //    var z = scope.Register(DeZero.NET.Functions.Sigmoid.Invoke(gates[0])[0]);

        //    // Reset gate
        //    var r = scope.Register(DeZero.NET.Functions.Sigmoid.Invoke(gates[1])[0]);

        //    // New memory content
        //    var h_tilde = scope.Register(DeZero.NET.Functions.Tanh.Invoke(gates[2])[0]);

        //    // Output computation
        //    var ret = ComputeOutput(z, state, h_tilde);
        //    return ret;
        //}

        private (Variable, Variable, Variable) SplitGates(Variable gates, int hiddenSize)
        {
            var data = gates.Data.Value;
            return (
                data.Slice(new[] { new Slice(), new Slice(0, hiddenSize) }).ToVariable(),
                data.Slice(new[] { new Slice(), new Slice(hiddenSize, 2 * hiddenSize) }).ToVariable(),
                data.Slice(new[] { new Slice(), new Slice(2 * hiddenSize, 3 * hiddenSize) }).ToVariable()
            );
        }

        private Variable[] ComputeGatesFP16(Variable x, Variable weights, Variable state)
        {
            using var scope = new ComputationScope();

            var fp16X = ConvertToFP16(x);
            var fp16Weights = ConvertToFP16(weights);
            var fp16State = ConvertToFP16(state);

            var gateOutputs = BatchedGEMM(fp16X, fp16Weights, fp16State);
            return gateOutputs.Select(g => ConvertToFP32(g)).ToArray();
        }

        private Variable[] ComputeGatesFP32(Variable x, Variable weights, Variable state)
        {
            return BatchedGEMM(x, weights, state);
        }

        private Variable ComputeOutput(Variable z, Variable state, Variable h_tilde)
        {
            using var scope = new ComputationScope();

            var zState = scope.Register(DeZero.NET.Functions.Mul.Invoke(z, state)[0]);
            var oneMinusZ = scope.Register(DeZero.NET.Functions.Sub.Invoke(new NDarray(1).ToVariable(), z)[0]);
            var zNewState = scope.Register(DeZero.NET.Functions.Mul.Invoke(oneMinusZ, h_tilde)[0]);

            var ret = DeZero.NET.Functions.Add.Invoke(zState, zNewState).Item1[0];
            return ret;
        }

        private Variable PrecomputeWeightMatrices()
        {
            // 事前に正しい形状の重み行列を作成
            var inputSize = Wxz.Value.W.Value.Shape[1];
            var hiddenSize = Whz.Value.OutSize.Value;

            var combinedShape = new Shape(inputSize + hiddenSize, 3 * hiddenSize);
            var wxPart = DeZero.NET.Functions.Concatenate.Invoke(new[] {
                Wxz.Value.W.Value,
                Wxr.Value.W.Value,
                Wxh.Value.W.Value
            }, axis: 1)[0];

            var whPart = DeZero.NET.Functions.Concatenate.Invoke(new[] {
                Whz.Value.W.Value,
                Whr.Value.W.Value,
                Whh.Value.W.Value
            }, axis: 1)[0];

            return DeZero.NET.Functions.Concatenate.Invoke(wxPart, whPart, axis: 0)[0];
        }
        //private Variable PrecomputeWeightMatrices()
        //{
        //    //using var scope = new ComputationScope();

        //    // 各行列の形状を確認してから結合
        //    var wxCombined = CombineMatrices(
        //        Wxz.Value.W.Value.Data.Value,
        //        Wxr.Value.W.Value.Data.Value,
        //        Wxh.Value.W.Value.Data.Value
        //    );
        //    var whCombined = CombineMatrices(
        //        Whz.Value.W.Value.Data.Value,
        //        Whr.Value.W.Value.Data.Value,
        //        Whh.Value.W.Value.Data.Value
        //    );

        //    // axis=0で結合（行方向）
        //    return DeZero.NET.Functions.Concatenate.Invoke(wxCombined, whCombined, axis: 0)[0];
        //}


        private Variable GetOptimizedWeights()
        {
            if (!EnablePrefetch) return PrecomputeWeightMatrices();

            return _precomputedWeights[_currentBatchIndex % PREFETCH_BUFFER_SIZE];
        }

        private Variable GetOrCreateState(int batchSize)
        {
            if (H.Value?.Data?.Value is null || H.Value?.Shape is null || H.Value?.Shape[0] != batchSize)
            {
                H.Value?.Dispose();
                H.Value = xp.zeros(new Shape(batchSize, Whz.Value.OutSize.Value)).ToVariable();
            }
            return H.Value;
        }

        private Variable GetCachedState(int startIdx, int endIdx)
        {
            return _stateCache.GetOrAdd(startIdx, _ =>
                xp.zeros(new Shape(endIdx - startIdx, Whz.Value.OutSize.Value)).ToVariable());
        }

        private void UpdateState(Variable newState)
        {
            lock (_stateLock)
            {
                H.Value?.Dispose();
                H.Value = newState;
                _currentBatchIndex++;

                if (_currentBatchIndex % CACHE_CLEANUP_INTERVAL == 0)
                {
                    CleanupCache();
                }
            }
        }

        private Variable CombineResults(Variable[] results)
        {
            try
            {
                return DeZero.NET.Functions.Concatenate.Invoke(results, axis: 0)[0];
            }
            finally
            {
                foreach (var result in results)
                {
                    result?.Dispose();
                }
            }
        }

        private void CleanupCache()
        {
            foreach (var state in _stateCache.Values)
            {
                state?.Dispose();
            }
            _stateCache.Clear();
            _weightCache.Clear();
            GC.Collect(1, GCCollectionMode.Optimized);
        }

        private void ThrowIfDisposed()
        {
            if (_isDisposed)
            {
                throw new ObjectDisposedException(nameof(OptimizedGRU));
            }
        }

        public void Dispose()
        {
            if (_isDisposed) return;
            _isDisposed = true;

            H.Value?.Dispose();
            _weightCache.Dispose();

            foreach (var weight in _precomputedWeights)
            {
                weight?.Dispose();
            }

            CleanupCache();
        }

        // Helper methods for FP16 conversion and GEMM operations
        private Variable ConvertToFP16(Variable x)
        {
            return x.Data.Value.astype(Dtype.float16).ToVariable();
        }

        private Variable ConvertToFP32(Variable x)
        {
            return x.Data.Value.astype(Dtype.float32).ToVariable();
        }

        private Variable[] BatchedGEMM(Variable x, Variable weights, Variable state)
        {
            var hiddenSize = Whz.Value.OutSize.Value;

            // メモリ割り当てを最小化
            var xh = DeZero.NET.Functions.Concatenate.Invoke(x, state, axis: 1)[0];
            var result = DeZero.NET.Functions.MatMul.Invoke(
                xh,
                weights.Data.Value.reshape(new Shape(-1, 3 * hiddenSize)).ToVariable()
            )[0];

            // メモリコピーを最小限に
            var data = result.Data.Value;
            return new[] {
                data.Slice(new[] { new Slice(), new Slice(0, hiddenSize) }).ToVariable(),
                data.Slice(new[] { new Slice(), new Slice(hiddenSize, 2 * hiddenSize) }).ToVariable(),
                data.Slice(new[] { new Slice(), new Slice(2 * hiddenSize, 3 * hiddenSize) }).ToVariable()
            };
        }
        //private Variable[] BatchedGEMM(Variable x, Variable weights, Variable state)
        //{
        //    using var scope = new ComputationScope();

        //    var xh = DeZero.NET.Functions.Concatenate.Invoke(x, state, axis: 1)[0];
        //    // weightsの形状を(72, 192)に変更
        //    var reshapedWeights = weights.Data.Value.reshape(new Shape(72, 3 * Whz.Value.OutSize.Value)).ToVariable();

        //    var result = DeZero.NET.Functions.MatMul.Invoke(xh, reshapedWeights)[0];

        //    // Split result into three parts for the three gates
        //    var hiddenSize = Whz.Value.OutSize.Value;
        //    var splits = SplitTensor(result, hiddenSize);

        //    return splits;
        //}

        private Variable[] SplitTensor(Variable tensor, int hiddenSize)
        {
            var data = tensor.Data.Value;
            var shape = tensor.Shape;

            return new[]
            {
                data.Slice(new[] { new Slice(), new Slice(0, hiddenSize) }).ToVariable(),
                data.Slice(new[] { new Slice(), new Slice(hiddenSize, 2 * hiddenSize) }).ToVariable(),
                data.Slice(new[] { new Slice(), new Slice(2 * hiddenSize, 3 * hiddenSize) }).ToVariable()
            };
        }

        private Variable CombineMatrices(params NDarray[] matrices)
        {
            // 結合前に各行列の形状を確認
            return DeZero.NET.Functions.Concatenate.Invoke(
                matrices.Select(m => m.ToVariable()).ToArray(),
                axis: 0
            )[0];
        }

        private sealed class BatchComputeScope : IDisposable
        {
            private readonly List<IDisposable> _resources = new();

            public void Register(IDisposable resource)
            {
                _resources.Add(resource);
            }

            public void Dispose()
            {
                foreach (var resource in _resources)
                {
                    resource?.Dispose();
                }
                _resources.Clear();
            }
        }

        public void ClearCache()
        {
            lock (_stateLock)
            {
                CleanupCache();
                _currentBatchIndex = 0;
            }
        }

        public void ResetState()
        {
            lock (_stateLock)
            {
                H.Value?.Dispose();
                H.Value = null;

                foreach (var state in _stateCache.Values)
                {
                    state?.Dispose();
                }
                _stateCache.Clear();
                _currentBatchIndex = 0;
            }
        }
    }
}