using DeZero.NET;
using DeZero.NET.Core;
using DeZero.NET.Extensions;
using DeZero.NET.Layers.Recurrent;
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
        private Variable _lastValidGRUState = null;
        private const int MEMORY_CHECK_INTERVAL = 5; // メモリチェックの間隔
        private readonly MemoryWatcher _memoryWatcher;
        private const int CLEANUP_INTERVAL = 5;
        private int _previousMaxMemory = 0;
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
        private bool isDebugMode = false;
        private Queue<Variable> stateQueue;
        private Queue<string> diagnosticsQueue;

        public DCNNModel()
        {
            float width_mult = 0.5f;
            int mobilenet_output_channels = 32;
            _memoryWatcher = new MemoryWatcher(MAX_MEMORY_THRESHOLD);

            Cnn = new MobileNet(mobilenet_output_channels, width_mult);
            int gru_input_size = mobilenet_output_channels * 1 * 1;
            int gru_hidden_size = 256;

            Gru1 = new GRU(gru_input_size, gru_hidden_size);
            Fc1 = new L.Linear.Linear(128, in_size: gru_hidden_size);
            Fc2 = new L.Linear.Linear(64, in_size: 128);
            Fc3 = new L.Linear.Linear(3, in_size: 64);

            SetAttribute("MobileNet", Cnn);
            SetAttribute("GRU1", Gru1);
            SetAttribute("Fc1", Fc1);
            SetAttribute("Fc2", Fc2);
            SetAttribute("Fc3", Fc3);

            stateQueue = new Queue<Variable>();
            diagnosticsQueue = new Queue<string>();
            ResetState();
        }

        public override Variable[] Forward(params Variable[] inputs)
        {
            using var scope = new BatchScope();
            var diagnosticsLog = new System.Text.StringBuilder();
            try
            {
                var result = ProcessForwardPass(inputs, scope, diagnosticsLog);
                _memoryWatcher.CheckAndCleanup(_forwardCallCount);
                return result;
            }
            catch (Exception ex)
            {
                HandleError(ex, diagnosticsLog);
                return CreateZeroOutput();
            }
            finally
            {
                scope.Dispose();
                if (_forwardCallCount % CLEANUP_INTERVAL == 0)
                {
                    ForceCleanup();
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
                Console.WriteLine("Invalid input to Forward pass");
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

            if (!cnnValid)
            {
                Console.WriteLine("CNN validation failed");
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

                using var memInfo = new GpuMemoryInfo();
                if (memInfo.UsedMemoryMB > MAX_MEMORY_THRESHOLD)
                {
                    ResetGRUState();
                    GpuMemoryMonitor.ForceMemoryPool();
                }

                var gruOutput = Gru1.Forward(x)[0];
                var (validatedOutput, isValid, diag) = StabilizeAndValidate(gruOutput, "GRU");
                diagnosticsLog.AppendLine(diag);

                if (!isValid)
                {
                    return stateQueue.Count > 0 ? stateQueue.Peek() : CreateZeroState(x.Shape[0]);
                }

                ManageGRUState(validatedOutput, diag);

                Variable reshapedOutput;
                if (validatedOutput.Shape.Dimensions.Length == 3)
                {
                    reshapedOutput = DeZero.NET.Functions.Reshape.Invoke(validatedOutput,
                        new Shape(validatedOutput.Shape[0], -1))[0];
                    scope.TrackTemporary(validatedOutput);
                }
                else
                {
                    reshapedOutput = validatedOutput;
                }

                if (_forwardCallCount % GRU_MEMORY_CLEANUP_INTERVAL == 0)
                {
                    CleanupGRUMemory();
                }

                return reshapedOutput;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"GRU forward error: {ex.Message}");
                return stateQueue.Count > 0 ? stateQueue.Peek() : CreateZeroState(x.Shape[0]);
            }
        }

        //private Variable ProcessGRUForward(Variable x, BatchScope scope, StringBuilder diagnosticsLog)
        //{
        //    try
        //    {
        //        GpuMemoryMonitor.Instance.LogMemoryUsage("Before GRU");

        //        // 入力を追跡
        //        scope.TrackTemporary(x);

        //        // メモリ使用量チェックと状態リセット
        //        using (var memInfo = new GpuMemoryInfo())
        //        {
        //            if (memInfo.UsedMemoryMB > MAX_MEMORY_THRESHOLD)
        //            {
        //                ResetGRUState();
        //                GpuMemoryMonitor.ForceMemoryPool();
        //            }

        //            var gruOutput = Gru1.Forward(x)[0];
        //            var (validatedOutput, isValid, diag) = StabilizeAndValidate(gruOutput, "GRU");
        //            diagnosticsLog.AppendLine(diag);

        //            if (!isValid)
        //            {
        //                return _lastValidGRUState ?? CreateZeroState(x.Shape[0]);
        //            }

        //            // 状態を管理
        //            ManageGRUState(validatedOutput, diag);

        //            // 前の状態をクリーンアップ
        //            if (_lastValidGRUState != null)
        //            {
        //                _lastValidGRUState.Dispose();
        //            }

        //            // 新しい状態のコピーを保存
        //            _lastValidGRUState = validatedOutput.Data.Value.copy().ToVariable();

        //            // GRU出力の形状変換
        //            Variable reshapedOutput;
        //            if (validatedOutput.Shape.Dimensions.Length == 3)
        //            {
        //                reshapedOutput = DeZero.NET.Functions.Reshape.Invoke(validatedOutput,
        //                    new Shape(validatedOutput.Shape[0], -1))[0];
        //                scope.TrackTemporary(validatedOutput); // 元の出力を追跡
        //            }
        //            else
        //            {
        //                reshapedOutput = validatedOutput;
        //            }

        //            // 定期的なクリーンアップ
        //            if (_forwardCallCount % GRU_MEMORY_CLEANUP_INTERVAL == 0)
        //            {
        //                CleanupGRUMemory();
        //            }

        //            return reshapedOutput;
        //        }
        //    }
        //    catch (Exception ex)
        //    {
        //        Console.WriteLine($"GRU forward error: {ex.Message}");
        //        return _lastValidGRUState ?? CreateZeroState(x.Shape[0]);
        //    }
        //}

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
                if (x == null)
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
                Console.WriteLine($"Error in ManageGRUState: {ex.Message}");
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
            Console.WriteLine(diagnosticsLog.ToString());
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