using Cupy.Log;
using DeZero.NET;
using DeZero.NET.Core;
using DeZero.NET.Datasets;
using DeZero.NET.Extensions;
using DeZero.NET.Functions;
using DeZero.NET.LearningRateSchedulers;
using DeZero.NET.Models;
using DeZero.NET.Optimizers;
using MHWGoldCrownModelTrainWorker;
using MovieFileDataLoaderSampleWorker;
using Python.Runtime;


var globalLogLevel = DeZero.NET.Log.LogLevel.Info;
var globalVerbose = false;

Cupy.Utils.GpuMemoryMonitor.LogLevel = LogLevel.Info;
Cupy.Utils.VRAMLeakDetector.IsEnabled = false;
//Cupy.Utils.PythonObjectTracker.DebugDetectingShape = "(32, 224, 224, 3)";

GpuMemoryMonitor.IsVerbose = false;
GpuMemoryMonitor.LogLevel = DeZero.NET.Log.LogLevel.Info;

var workerProcess = new WorkerProcess();

workerProcess.SetTrainSet(() => new GoldCrownTrainDataset());
workerProcess.SetTestSet(() => new GoldCrownTestDataset());
workerProcess.SetTrainLoader((ts, batch_size) => new MovieFileDataLoader((MovieFileDataset)ts, workerProcess.BatchSize, () =>
{
    (workerProcess.Model as DCNNModel).ResetState();
    workerProcess.SaveWeights();
    workerProcess.SaveOptimizer();
}, shuffle: false));
workerProcess.SetTestLoader((ts, batch_size) => new MovieFileDataLoader((MovieFileDataset)ts, workerProcess.BatchSize, () => { (workerProcess.Model as DCNNModel).ResetState(); }, shuffle: false));


//workerProcess.SetTrainSet(() => new SampleMovieFileDataset());
//workerProcess.SetTestSet(() => new SampleMovieFileDataset());
//workerProcess.SetTrainLoader((ts, batch_size) => new SampleDataLoader((MovieFileDataset)ts, workerProcess.BatchSize, () =>
//{
//    (workerProcess.Model as DCNNModel).ResetState();
//    workerProcess.SaveWeights();
//    workerProcess.SaveOptimizer();
//}, shuffle: false));
//workerProcess.SetTestLoader((ts, batch_size) => new SampleDataLoader((MovieFileDataset)ts, workerProcess.BatchSize, () => { (workerProcess.Model as DCNNModel).ResetState(); }, shuffle: false));

workerProcess.SetModel(() => new DCNNModel(isVerbose: globalVerbose, logLevel: globalLogLevel));
workerProcess.LoadExistedWeights();
workerProcess.SetOptimizer(model => new AdamW().Setup(model));
workerProcess.SetLearningRateScheduler(() => new ReduceLROnPlateau(
                                                19 * 20,
                                                initialLr: 0.1f,
                                                minLr: 0.0000001f,  // より広い範囲で探索
                                                maxLr: 0.1f         // 発散を防ぐ
                                            ), 0.001f);
workerProcess.LoadOptimizer();
workerProcess.InitializeLossPlotter(100000);
workerProcess.ResumeState();


using dynamic cupy = Py.Import("cupy");
using dynamic mempool = cupy.get_default_memory_pool();
mempool.set_limit(1024L * 1024 * 1024 * 22);

workerProcess.Run();

class WorkerProcess : DeZero.NET.Processes.WorkerProcess
{
    public WorkerProcess() : base(DeZero.NET.Log.LogLevel.Debug)
    {
    }

    public override string PythonDLLPath => @"C:\Users\boiler\AppData\Local\Programs\Python\Python311\python311.dll";
    protected override void InitializeArguments(object[] args)
    {
        Epoch = int.Parse(args[0].ToString());
        BatchSize = int.Parse(args[1].ToString());
        HiddenSize = int.Parse(args[2].ToString());
        EnableGpu = bool.Parse(args[3].ToString());
        RecordFilePath = args[4].ToString().Replace("'", string.Empty);
        DisposeAllInputs = true;
    }

    // 勾配クリッピングのための閾値
    private const float GRAD_CLIP_THRESHOLD = 1.0f;

    public override Variable CalcAdditionalLoss(Variable loss)
    {
        using var scope = this.TrackMemory("CalcAdditionalLoss");
        try
        {
            using var ooo1 = new NDarray(0.001f).ToVariable();
            using var l2r = L2Regularization.Invoke(Model.Params(), ooo1)[0];
            // L2正則化の強度を調整
            return loss + l2r;
        }
        finally
        {
        }
    }

    // 正規化された損失を計算する補助メソッド
    private Variable CalculateNormalizedLoss(Variable y, NDarray t)
    {
        using var _ = this.TrackMemory("CalculateNormalizedLoss");
        using var scope = new ComputationScope();
        var normalized_losses = new List<Variable>();
        try
        {
            using var y_shape = y.Shape;
            var batch_size = y_shape[0];
            var feature_dim = y_shape[1];

            for (int dim = 0; dim < feature_dim; dim++)
            {
                // Variableとして平均と標準偏差を計算
                // 次元方向の平均
                using var y_mean = DeZero.NET.Functions.Mean.Invoke(y, axis: 1, keepdims: true)[0];
                using var _0 = y - y_mean;
                using var a = DeZero.NET.Functions.Square.Invoke(_0)[0];
                using var b = DeZero.NET.Functions.Mean.Invoke(a, axis: 1, keepdims: true)[0];
                using var c = b + 1e-8f;
                using var y_std = DeZero.NET.Functions.Sqrt.Invoke(c)[0];

                // 正規化 - すべてVariableの演算として実行
                using var d = y - y_mean;
                using var normalized_y = d / y_std;

                // tをVariableに変換（一度だけ）
                var t_variable = t.ToVariable();
                using var t_mean = DeZero.NET.Functions.Mean.Invoke(t_variable, axis: 1, keepdims: true)[0];
                using var _1 = t_variable - t_mean;
                using var e = DeZero.NET.Functions.Square.Invoke(_1)[0];
                using var f = DeZero.NET.Functions.Mean.Invoke(e, axis: 1, keepdims: true)[0];
                using var g = f + 1e-8f;
                using var t_std = DeZero.NET.Functions.Sqrt.Invoke(g)[0];
                using var normalized_t = _1 / t_std;

                // Huber Lossの計算
                using var diff = normalized_y - normalized_t;
                using var abs_diff = DeZero.NET.Functions.Abs.Invoke(diff)[0];
                using var delta = DeZero.NET.Functions.Const.Invoke(1.0f)[0];

                // Huber Loss - すべてFunctionsを使用
                using var i = DeZero.NET.Functions.Square.Invoke(diff)[0];
                using var j = Const.Invoke(0.5f)[0];
                using var quadratic = DeZero.NET.Functions.Mul.Invoke(j, i)[0];
                using var k = DeZero.NET.Functions.Mul.Invoke(delta[0], abs_diff[0])[0];
                using var l_a = DeZero.NET.Functions.Const.Invoke(0.5f)[0];
                using var l_b = DeZero.NET.Functions.Square.Invoke(delta[0])[0];
                using var l = DeZero.NET.Functions.Mul.Invoke(l_a, l_b)[0];
                using var linear = DeZero.NET.Functions.Sub.Invoke(k, l)[0];

                using var condition = DeZero.NET.Functions.LessThan.Invoke(abs_diff, delta).Item1[0];
                using var huber_loss = DeZero.NET.Functions.Where.Invoke(condition, quadratic, linear).Item1[0];

                // スケーリング係数もVariableとして計算
                using var scaling_factor = DeZero.NET.Functions.Div.Invoke(t_std, y_std)[0];
                var scaled_loss = DeZero.NET.Functions.Mul.Invoke(huber_loss, scaling_factor)[0];

                normalized_losses.Add(scaled_loss);
            }

            // 損失の結合
            var combined_loss = normalized_losses[0];
            for (int i = 1; i < normalized_losses.Count; i++)
            {
                scope.Register(combined_loss);
                scope.Register(normalized_losses[i]);
                combined_loss = DeZero.NET.Functions.Add.Invoke(combined_loss, normalized_losses[i]).Item1[0];
            }

            // バッチ全体の平均
            using var total_loss = DeZero.NET.Functions.Sum.Invoke(combined_loss)[0];
            using var m = Const.Invoke(batch_size * feature_dim)[0];
            var mean_loss = DeZero.NET.Functions.Div.Invoke(total_loss, m)[0];

            return mean_loss;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error in normalized loss calculation: {ex.Message}");
            throw;
        }
        finally
        {
            foreach (var loss in normalized_losses)
            {
                loss?.Dispose();
            }
        }
    }

    public override Variable CalcLoss(Variable y, NDarray t)
    {
        using var scope = this.TrackMemory("CalcLoss");
        try
        {
            var loss = CalculateNormalizedLoss(y, t);

            // トレーニングフェーズでのみ勾配クリッピングを試みる
            if (Config.EnableBackprop)
            {
                // 一時的な変数を作成して勾配を計算
                using var temp_loss = loss.Data.Value.copy().ToVariable();
                temp_loss.Backward(retain_grad: true);

                // 勾配が存在する場合のみクリッピングを適用
                if (temp_loss.Grad != null)
                {
                    using var grad_norm = xp.linalg.norm(temp_loss.Grad.Value.Data.Value);
                    if (grad_norm.asscalar<float>() != float.NaN && grad_norm.asscalar<float>() > GRAD_CLIP_THRESHOLD)
                    {
                        var scale = GRAD_CLIP_THRESHOLD / (grad_norm.asscalar<float>() + 1e-8f);
                        loss = new Variable(loss.Data.Value);
                        loss.Grad.Value = new Variable(temp_loss.Grad.Value.Data.Value * scale);
                    }
                }
            }

            return loss;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error in loss calculation: {ex.Message}");
            throw;
        }
        finally
        {
        }
    }

    public override Variable CalcEvaluationMetric(Variable y, NDarray t)
    {
        using var scope = this.TrackMemory("CalcEvaluationMetric");
        try
        {
            using var y_shape = y.Shape;
            // 各次元の平均絶対誤差を計算
            var batch_size = y_shape[0];
            var feature_dim = y_shape[1];

            using var diff = y - t.ToVariable();
            using var abs_diff = Abs.Invoke(diff)[0];

            using var sum = Sum.Invoke(abs_diff)[0];

            // すべての次元の誤差を合計し、サンプル数と次元数で割って平均を取る
            return sum / (batch_size * feature_dim);
        }
        finally
        {
        }
    }

    protected override Func<NDarray, long> UnitLength => (t) => 1;

    public override ModelType ModelType => ModelType.Regression;
}
