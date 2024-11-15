using DeZero.NET;
using DeZero.NET.Extensions;
using DeZero.NET.Layers.Recurrent;
using DeZero.NET.Models;
using Python.Runtime;
using L = DeZero.NET.Layers;

namespace MovieFileDataLoaderSampleWorker
{
    class DCNNModel : Model
    {
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

        private void ManageSequenceState(Variable gru1Out, string diagnosticInfo)
        {
            stateQueue.Enqueue(gru1Out);
            diagnosticsQueue.Enqueue(diagnosticInfo);

            if (stateQueue.Count > SEQUENCE_LENGTH)
            {
                stateQueue.Dequeue();
                diagnosticsQueue.Dequeue();

                // シーケンス長に達した場合、状態をリセットして再計算
                if (stateQueue.Count == SEQUENCE_LENGTH)
                {
                    ResetState();
                    var tempStates = stateQueue.ToList();
                    var tempDiagnostics = diagnosticsQueue.ToList();
                    stateQueue.Clear();
                    diagnosticsQueue.Clear();

                    foreach (var (state, diag) in tempStates.Zip(tempDiagnostics, (s, d) => (s, d)))
                    {
                        var (validatedState, isValid, _) = StabilizeAndValidate(state, "GRU-Recompute");
                        if (isValid)
                        {
                            Gru1.Forward(validatedState);
                            stateQueue.Enqueue(validatedState);
                            diagnosticsQueue.Enqueue(diag);
                        }
                    }
                }
            }
        }

        public void ResetState()
        {
            Gru1.ResetState();
            stateQueue.Clear();
            diagnosticsQueue.Clear();
        }

        public override Variable[] Forward(params Variable[] inputs)
        {
            //using var scope = new MemoryManagementExtensions.BatchScope();
            var diagnosticsLog = new System.Text.StringBuilder();
            try
            {
                if (inputs == null || inputs.Length == 0 || inputs[0] == null)
                {
                    throw new ArgumentException("Invalid input to Forward pass");
                }

                var x = inputs[0];
                //scope.RegisterForDisposal(x);
                expectedOutputShape = new Shape(x.Shape[0], 3);

                // CNN forward pass
                var (cnnOutput, cnnValid, cnnDiag) = StabilizeAndValidate(Cnn.Forward(x)[0], "CNN");
                diagnosticsLog.AppendLine(cnnDiag);
                if (!cnnValid)
                {
                    Console.WriteLine(diagnosticsLog.ToString());
                    return new[] { new Variable(xp.zeros(expectedOutputShape)) };
                }
                x = cnnOutput;
                //scope.RegisterForDisposal(cnnOutput);
                GC.Collect();

                // Reshape CNN output
                var flattenedShape = new Shape(x.Shape[0], -1);
                x = DeZero.NET.Functions.Reshape.Invoke(x, flattenedShape)[0];
                var (reshapedOutput, reshapeValid, reshapeDiag) = StabilizeAndValidate(x, "CNN Reshape");
                diagnosticsLog.AppendLine(reshapeDiag);
                if (!reshapeValid)
                {
                    Console.WriteLine(diagnosticsLog.ToString());
                    return new[] { new Variable(xp.zeros(expectedOutputShape)) };
                }
                x = reshapedOutput;
                //scope.RegisterForDisposal(reshapedOutput);

                // GRU forward pass with sequence management
                var (gruOutput, gruValid, gruDiag) = StabilizeAndValidate(Gru1.Forward(x)[0], "GRU");
                diagnosticsLog.AppendLine(gruDiag);
                if (!gruValid)
                {
                    Console.WriteLine(diagnosticsLog.ToString());
                    return new[] { new Variable(xp.zeros(expectedOutputShape)) };
                }
                var gru1Out = gruOutput;
                //scope.RegisterForDisposal(gruOutput);
                GC.Collect();

                // シーケンス状態の管理
                ManageSequenceState(gru1Out, gruDiag);

                // Reshape GRU output if needed
                if (gru1Out.Shape.Dimensions.Length == 3)
                {
                    gru1Out = DeZero.NET.Functions.Reshape.Invoke(gru1Out, new Shape(gru1Out.Shape[0], -1))[0];
                }

                // FC layers with ReLU
                var (fc1Output, fc1Valid, fc1Diag) = StabilizeAndValidate(Fc1.Forward(gru1Out)[0], "FC1", true);
                diagnosticsLog.AppendLine(fc1Diag);
                if (!fc1Valid)
                {
                    Console.WriteLine(diagnosticsLog.ToString());
                    return new[] { new Variable(xp.zeros(expectedOutputShape)) };
                }
                x = fc1Output;
                //scope.RegisterForDisposal(fc1Output);
                GC.Collect();

                var (fc2Output, fc2Valid, fc2Diag) = StabilizeAndValidate(Fc2.Forward(x)[0], "FC2", true);
                diagnosticsLog.AppendLine(fc2Diag);
                if (!fc2Valid)
                {
                    Console.WriteLine(diagnosticsLog.ToString());
                    return new[] { new Variable(xp.zeros(expectedOutputShape)) };
                }
                x = fc2Output;
                //scope.RegisterForDisposal(fc2Output);
                GC.Collect();

                var (fc3Output, fc3Valid, fc3Diag) = StabilizeAndValidate(Fc3.Forward(x)[0], "FC3");
                diagnosticsLog.AppendLine(fc3Diag);
                if (!fc3Valid)
                {
                    Console.WriteLine(diagnosticsLog.ToString());
                    return new[] { new Variable(xp.zeros(expectedOutputShape)) };
                }
                x = fc3Output;
                GC.Collect();

                // 出力の最終チェック
                if (x.Shape.Dimensions.Length != 2 || x.Shape[1] != 3)
                {
                    diagnosticsLog.AppendLine($"Invalid output shape: {string.Join("x", x.Shape.Dimensions)}");
                    Console.WriteLine(diagnosticsLog.ToString());
                    return new[] { new Variable(xp.zeros(expectedOutputShape)) };
                }

                if (isDebugMode)
                {
                    Console.WriteLine(diagnosticsLog.ToString());
                }

                return new[] { x };
            }
            catch (Exception ex)
            {
                diagnosticsLog.AppendLine($"Forward pass error: {ex.Message}");
                Console.WriteLine(diagnosticsLog.ToString());
                return new[] { new Variable(xp.zeros(expectedOutputShape)) };
            }
            finally
            {
                GC.Collect();
                Finalizer.Instance.Collect();
            }
        }

        public void InitializeLSTMStates(int batch_size)
        {
            ResetState();
        }
    }
}