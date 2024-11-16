using DeZero.NET;
using DeZero.NET.Core;
using DeZero.NET.Extensions;
using DeZero.NET.Layers.Recurrent;
using DeZero.NET.Log;
using DeZero.NET.Models;
using Python.Runtime;
using System.Text;
using L = DeZero.NET.Layers;

namespace MovieFileDataLoaderSampleWorker
{
    class DCNNModel : Model
    {
        private readonly int GRU_MEMORY_CLEANUP_INTERVAL = 10;
        private readonly int MAX_MEMORY_THRESHOLD = 100; // MB
        private readonly int BATCH_PROCESSING_SIZE = 128; // バッチ処理サイズを制限
        private Variable _lastValidGRUState = null;
        private readonly MemoryWatcher _memoryWatcher;
        private int _forwardCallCount = 0;
        public MobileNet Cnn { get; set; }
        public GRU Gru1 { get; set; }
        public L.Linear.Linear Fc1 { get; set; }
        public L.Linear.Linear Fc2 { get; set; }
        public L.Linear.Linear Fc3 { get; set; }

        private const float EPSILON = 1e-8f;
        private const float MAX_VALUE = 1e6f;
        private const int SEQUENCE_LENGTH = 60; // シーケンス長の制限
        private Shape expectedOutputShape;
        private Queue<Variable> stateQueue;
        private Queue<string> diagnosticsQueue;

        private readonly bool _isVerbose;
        private readonly LogLevel _logLevel;
        private readonly ILogger _logger;

        public DCNNModel(bool isVerbose = false, LogLevel logLevel = LogLevel.Error)
        {
            _isVerbose = isVerbose;
            _logLevel = logLevel;
            _logger = new ConsoleLogger(logLevel, isVerbose);

            _logger.LogDebug("Initializing DCNNModel");

            float width_mult = 0.25f; // Reduced from 0.5f
            int mobilenet_output_channels = 16; // Reduced from 32

            _logger.LogDebug($"Creating MobileNet with width_mult={width_mult}, output_channels={mobilenet_output_channels}");
            Cnn = new MobileNet(mobilenet_output_channels, width_mult);
            int gru_input_size = 16;
            int gru_hidden_size = 128;

            _logger.LogDebug($"Creating GRU with input_size={gru_input_size}, hidden_size={gru_hidden_size}");
            Gru1 = new GRU(gru_input_size, gru_hidden_size, isVerbose, logLevel);
            ConfigureGRUOptimizations();

            _logger.LogDebug("Creating Fully Connected layers");
            Fc1 = new L.Linear.Linear(128, in_size: gru_hidden_size);
            Fc2 = new L.Linear.Linear(64, in_size: 128);
            Fc3 = new L.Linear.Linear(3, in_size: 64);

            SetAttribute("MobileNet", Cnn);
            SetAttribute("GRU1", Gru1);
            SetAttribute("Fc1", Fc1);
            SetAttribute("Fc2", Fc2);
            SetAttribute("Fc3", Fc3);

            _memoryWatcher = new MemoryWatcher(85);
            stateQueue = new Queue<Variable>();
            diagnosticsQueue = new Queue<string>();
            ResetState();

            _logger.LogInfo("DCNNModel initialization completed");
        }

        private void ConfigureGRUOptimizations()
        {
            _logger.LogDebug("Configuring GRU optimizations");

            // GRUの計算最適化設定
            Gru1.EnableStateCompression = true; // 状態圧縮を有効化
            _logger.LogDebug("State compression enabled");

            Gru1.BatchProcessingEnabled = true;
            Gru1.BatchSize = BATCH_PROCESSING_SIZE;
            _logger.LogDebug($"Batch processing enabled with size {BATCH_PROCESSING_SIZE}");

            // キャッシュ設定
            Gru1.EnableWeightCaching = true;
            Gru1.CacheSize = 1000; // キャッシュサイズを制限
            _logger.LogDebug($"Weight caching enabled with cache size {Gru1.CacheSize}");
        }

        public override Variable[] Forward(params Variable[] inputs)
        {
            using var scope = new BatchScope();
            var diagnosticsLog = new StringBuilder();
            _logger.LogDebug("Starting Forward pass");

            try
            {
                // バッチ処理の最適化
                if (inputs[0].Shape[0] > BATCH_PROCESSING_SIZE)
                {
                    _logger.LogInfo($"Processing large batch of size {inputs[0].Shape[0]}");
                    return ProcessLargeBatch(inputs, scope);
                }

                var result = ProcessForwardPass(inputs, scope, diagnosticsLog);

                // メモリ管理の最適化
                if (_forwardCallCount % GRU_MEMORY_CLEANUP_INTERVAL == 0)
                {
                    _logger.LogDebug("Performing optimized cleanup");
                    PerformOptimizedCleanup();
                }

                _forwardCallCount++;
                return result;
            }
            catch (Exception ex)
            {
                _logger.LogError($"Error in Forward pass: {ex.Message}");
                HandleError(ex, diagnosticsLog);
                return CreateZeroOutput();
            }
        }

        private void PerformOptimizedCleanup()
        {
            using (var cleanupScope = new MemoryCleanupScope())
            {
                _logger.LogDebug("Starting memory cleanup");
                Gru1.ClearCache();
                _logger.LogTrace("GRU cache cleared");
                GpuMemoryMonitor.ForceMemoryPool();
                _logger.LogTrace("GPU memory pool forced");

                if (_memoryWatcher.IsMemoryHigh())
                {
                    _logger.LogWarning("High memory usage detected, performing full cleanup");
                    ResetGRUState();
                    GC.Collect(2, GCCollectionMode.Forced);
                }
                _logger.LogDebug("Memory cleanup completed");
            }
        }

        private Variable[] ProcessLargeBatch(Variable[] inputs, BatchScope scope)
        {
            var x = inputs[0];
            var batchSize = x.Shape[0];
            var results = new List<Variable>();

            // バッチを分割して処理
            for (int i = 0; i < batchSize; i += BATCH_PROCESSING_SIZE)
            {
                var endIdx = Math.Min(i + BATCH_PROCESSING_SIZE, batchSize);
                var batchSlice = x.Data.Value.Slice(new[] { new Slice(i, endIdx) });

                var batchResult = ProcessForwardPass(new[] { batchSlice.ToVariable(x) }, scope, new StringBuilder());
                results.Add(batchResult[0]);
            }

            // 結果を結合
            return new[] { ConcatenateResults(results) };
        }

        private Variable ConcatenateResults(List<Variable> results)
        {
            try
            {
                if (results == null || results.Count == 0)
                {
                    return CreateZeroOutput()[0];
                }

                // すべての結果が同じ形状を持っているか確認
                var firstShape = results[0].Shape;
                if (results.Any(r => !r.Shape.Dimensions.SequenceEqual(firstShape.Dimensions.Skip(1))))
                {
                    throw new InvalidOperationException("All results must have the same shape except for batch dimension");
                }

                // バッチ方向（axis=0）で結合
                var concatenatedResult = xp.concatenate(
                    results.Select(r => r.Data.Value).ToArray(),
                    axis: 0
                ).ToVariable();

                return concatenatedResult;
            }
            catch (Exception ex)
            {
                _logger.LogError($"Error concatenating results: {ex.Message}");
                return CreateZeroOutput()[0];
            }
            finally
            {
                // 元の結果を解放
                foreach (var result in results)
                {
                    result?.Dispose();
                }
            }
        }

        private Variable[] ProcessForwardPass(Variable[] inputs, BatchScope scope, StringBuilder diagnosticsLog)
        {
            if (!ValidateInputs(inputs, out var x)) return CreateZeroOutput();

            // Forward開始時に出力シェイプを設定
            expectedOutputShape = new Shape(x.Shape[0], 3);

            x = ProcessCNNForward(x, scope, diagnosticsLog);
            if (x == null) return CreateZeroOutput();

            x = ProcessGRUForward(x, scope, diagnosticsLog);
            if (x == null) return CreateZeroOutput();

            x = ProcessFCForward(x, scope, diagnosticsLog);
            if (x == null) return CreateZeroOutput();

            return new[] { x };
        }

        private bool ValidateInputs(Variable[] inputs, out Variable x)
        {
            x = null;
            if (inputs == null || inputs.Length == 0 || inputs[0] == null)
            {
                _logger.LogWarning("Invalid input to Forward pass");
                return false;
            }
            x = inputs[0];
            return true;
        }

        private Variable ProcessCNNForward(Variable x, BatchScope scope, StringBuilder diagnosticsLog)
        {
            GpuMemoryMonitor.Instance.LogMemoryUsage("Before CNN");
            var (cnnOutput, cnnValid, cnnDiag) = StabilizeAndValidate(Cnn.Forward(x)[0], "CNN");
            diagnosticsLog.AppendLine(cnnDiag);
            // CNNの出力直後にデバッグ出力を追加
            _logger.LogDebug($"CNN Output Shape: {string.Join(", ", cnnOutput.Shape.Dimensions)}");

            if (!cnnValid)
            {
                _logger.LogWarning("CNN validation failed");
                return null;
            }

            // CNNの出力は後続のGRUで必要なので、DisposeせずにスコープでTrackする
            scope.TrackTemporary(cnnOutput);
            GpuMemoryMonitor.Instance.LogMemoryUsage("After CNN");
            return cnnOutput;
        }

        private Variable ProcessGRUForward(Variable x, BatchScope scope, StringBuilder diagnosticsLog)
        {
            try
            {
                GpuMemoryMonitor.Instance.LogMemoryUsage("Before GRU");
                scope.TrackTemporary(x);

                // CNNの出力を適切な形状にリシェイプ
                _logger.LogDebug($"Before reshape: {string.Join(", ", x.Shape.Dimensions)}");
                var reshapedInput = DeZero.NET.Functions.Reshape.Invoke(x,
                    new Shape(x.Shape[0], x.Shape[1]))[0];
                _logger.LogDebug($"After reshape: {string.Join(", ", reshapedInput.Shape.Dimensions)}");

                using var memInfo = new GpuMemoryInfo();
                if (memInfo.UsedMemoryMB > MAX_MEMORY_THRESHOLD)
                {
                    ResetGRUState();
                    GpuMemoryMonitor.ForceMemoryPool();
                }
                // GRUの入力直前にデバッグ出力を追加
                _logger.LogDebug($"GRU Input Size: {Gru1.Wxz.Value.InSize.Value}");
                var gruOutput = Gru1.Forward(reshapedInput)[0];  // xからreshapedInputに変更
                var (validatedOutput, isValid, diag) = StabilizeAndValidate(gruOutput, "GRU");
                diagnosticsLog.AppendLine(diag);

                if (!isValid)
                {
                    return stateQueue.Count > 0 ? stateQueue.Peek() : CreateZeroState(x.Shape[0]);
                }

                ManageGRUState(validatedOutput, diag);
                scope.TrackTemporary(reshapedInput);  // 新しい行：reshapedInputのメモリ管理

                if (_forwardCallCount % GRU_MEMORY_CLEANUP_INTERVAL == 0)
                {
                    CleanupGRUMemory();
                }

                return validatedOutput;  // Variable reshapedOutputの除去
            }
            catch (Exception ex)
            {
                _logger.LogError($"GRU forward error: {ex.Message}");
                return stateQueue.Count > 0 ? stateQueue.Peek() : CreateZeroState(x.Shape[0]);
            }
        }

        private void CleanupGRUMemory()
        {
            GpuMemoryMonitor.ForceMemoryPool();
            GC.Collect();
            Finalizer.Instance.Collect();

            // キューの整理
            while (stateQueue.Count > SEQUENCE_LENGTH / 2)
            {
                var oldState = stateQueue.Dequeue();
                var oldDiag = diagnosticsQueue.Dequeue();
                oldState?.Dispose();
            }
        }

        private void ResetGRUState()
        {
            Gru1.ResetState();
            _lastValidGRUState?.Dispose();
            _lastValidGRUState = null;

            // キューのクリア
            while (stateQueue.Count > 0)
            {
                var state = stateQueue.Dequeue();
                var diag = diagnosticsQueue.Dequeue();
                state?.Dispose();
            }
        }

        private Variable CreateZeroState(int batchSize)
        {
            var hiddenSize = Gru1.Whz.Value.OutSize.Value; // GRUの隠れ層サイズ
            return new Variable(xp.zeros(new Shape(batchSize, hiddenSize)));
        }

        private Variable ProcessFCForward(Variable x, BatchScope scope, StringBuilder diagnosticsLog)
        {
            try
            {
                // FC1
                scope.TrackTemporary(x);
                var (fc1Output, fc1Valid, fc1Diag) = StabilizeAndValidate(Fc1.Forward(x)[0], "FC1", true);
                diagnosticsLog.AppendLine(fc1Diag);
                if (!fc1Valid) return null;

                scope.TrackTemporary(fc1Output);
                x = fc1Output;

                // メモリ使用量が閾値を超えた場合、中間データをクリア
                if (GpuMemoryMonitor.Instance.GetCurrentMemoryUsage() > MAX_MEMORY_THRESHOLD)
                {
                    GpuMemoryMonitor.ForceMemoryPool();
                }

                // FC2
                var (fc2Output, fc2Valid, fc2Diag) = StabilizeAndValidate(Fc2.Forward(x)[0], "FC2", true);
                diagnosticsLog.AppendLine(fc2Diag);
                if (!fc2Valid) return null;

                scope.TrackTemporary(fc2Output);
                x = fc2Output;

                // FC3
                var fc3Output = Fc3.Forward(x)[0];
                scope.TrackTemporary(x); // 前の出力をクリア

                return fc3Output;
            }
            finally
            {
                // 強制メモリ解放頻度を上げる
                if (_forwardCallCount % 2 == 0)
                {
                    GpuMemoryMonitor.ForceMemoryPool();
                    GC.Collect();
                }
            }
        }

        private (Variable output, bool isValid, string diagnosticInfo) StabilizeAndValidate(Variable x, string layerName, bool applyReLU = false)
        {
            try
            {
                if (x is null || x?.Data?.Value?.Array is null)
                {
                    return (null, false, $"{layerName}: Null input");
                }

                var data = x.Data.Value.flatten().GetData<float[]>();
                var diagnostics = new System.Text.StringBuilder();
                diagnostics.AppendLine($"{layerName} stats:");
                diagnostics.AppendLine($"Shape: {string.Join("x", x.Shape.Dimensions)}");

                if (data.Length > 0)
                {
                    var min = data.Min();
                    var max = data.Max();
                    var mean = data.Average();
                    var nonZeroCount = data.Count(v => Math.Abs(v) > EPSILON);

                    diagnostics.AppendLine($"Min: {min}, Max: {max}, Mean: {mean}");
                    diagnostics.AppendLine($"Non-zero values: {nonZeroCount}/{data.Length}");

                    var nanCount = data.Count(float.IsNaN);
                    var infCount = data.Count(float.IsInfinity);

                    if (nanCount > 0 || infCount > 0)
                    {
                        diagnostics.AppendLine($"Warning: Found {nanCount} NaN and {infCount} Inf values");
                        data = data.Select(v =>
                        {
                            if (float.IsNaN(v)) return 0.0f;
                            if (float.IsInfinity(v)) return Math.Sign(v) * MAX_VALUE;
                            return Math.Max(-MAX_VALUE, Math.Min(MAX_VALUE, v));
                        }).ToArray();

                        x = new Variable(xp.array(data).reshape(x.Shape));
                    }

                    var extremeValueCount = data.Count(v => Math.Abs(v) > MAX_VALUE / 2);
                    if (extremeValueCount > 0)
                    {
                        diagnostics.AppendLine($"Warning: Found {extremeValueCount} extreme values");
                        x = new Variable(xp.array(
                            data.Select(v => Math.Max(-MAX_VALUE / 2, Math.Min(MAX_VALUE / 2, v))).ToArray()
                        ).reshape(x.Shape));
                    }

                    if (applyReLU)
                    {
                        var preReLUNegatives = data.Count(v => v < 0);
                        diagnostics.AppendLine($"Pre-ReLU negative values: {preReLUNegatives}");
                        x = DeZero.NET.Functions.ReLU.Invoke(x)[0];
                    }

                    return (x, true, diagnostics.ToString());
                }

                return (x, false, $"{layerName}: Empty data");
            }
            catch (Exception ex)
            {
                return (x, false, $"{layerName} error: {ex.Message}");
            }
        }

        private void ManageGRUState(Variable gruOutput, string diagnosticInfo)
        {
            try
            {
                // 状態のコピーを作成
                var stateCopy = gruOutput.Data.Value.copy().ToVariable();
                stateQueue.Enqueue(stateCopy);
                diagnosticsQueue.Enqueue(diagnosticInfo);

                // シーケンス長を超えた場合の処理
                if (stateQueue.Count > SEQUENCE_LENGTH)
                {
                    // 最も古い状態を破棄
                    var oldState = stateQueue.Dequeue();
                    diagnosticsQueue.Dequeue();
                    oldState?.Dispose();

                    // シーケンス長に達した場合、状態を再計算
                    if (stateQueue.Count == SEQUENCE_LENGTH)
                    {
                        RecomputeGRUStates();
                    }
                }

                // メモリ管理
                using (var memInfo = new GpuMemoryInfo())
                {
                    if (memInfo.UsedMemoryMB > MAX_MEMORY_THRESHOLD)
                    {
                        CleanupGRUMemory();
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError($"Error in ManageGRUState: {ex.Message}");
                ResetGRUState();
            }
        }

        private void RecomputeGRUStates()
        {
            Gru1.ResetState();
            var tempStates = stateQueue.ToList();
            var tempDiagnostics = diagnosticsQueue.ToList();
            stateQueue.Clear();
            diagnosticsQueue.Clear();

            foreach (var state in tempStates)
            {
                var (validatedState, isValid, _) = StabilizeAndValidate(state, "GRU-Recompute");
                if (isValid)
                {
                    var newState = Gru1.Forward(validatedState)[0];
                    stateQueue.Enqueue(newState.Data.Value.copy().ToVariable());
                }
                state.Dispose();
            }
        }

        private Variable[] CreateZeroOutput()
        {
            return new[] { new Variable(xp.zeros(expectedOutputShape)) };
        }

        private void HandleError(Exception ex, StringBuilder diagnosticsLog)
        {
            diagnosticsLog.AppendLine($"Forward pass error: {ex.Message}");
            _logger.LogError(diagnosticsLog.ToString());
            GpuMemoryMonitor.Instance.LogMemoryUsage("Error in Forward Pass");
        }

        private void ForceCleanup()
        {
            Gru1.ResetState();
            _lastValidGRUState?.Dispose();
            _lastValidGRUState = null;
            GpuMemoryMonitor.ForceMemoryPool();
            GC.Collect();
            Finalizer.Instance.Collect();
        }

        public void ResetState()
        {
            Gru1.ResetState();
            stateQueue.Clear();
            diagnosticsQueue.Clear();
        }

        public void InitializeLSTMStates(int batch_size)
        {
            ResetState();
        }
    }
}