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
GpuMemoryMonitor.DefaultLogLevel = DeZero.NET.Log.LogLevel.Debug;
GpuMemoryMonitor.Mode = Mode.CaptureCupyNDarray;
//GpuMemoryMonitor.Filter = "";

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
workerProcess.InitializeLossPlotter();
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
        MaxEpoch = int.Parse(args[5].ToString());
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
            using var add = loss + l2r;
            return add.copy();
        }
        finally
        {
        }
    }

    // 正規化された損失を計算する補助メソッド
    private Variable CalculateNormalizedLoss(Variable y, NDarray t)
    {
        using var scope = new ComputationScope();

        var batch_size = y.Shape[0];

        // 1. モンスターの数（整数値）の損失
        using var y_count = SliceFunc.Invoke(y, [new Slice(), new Slice(0, 1, 1)])[0];// y.Data.Value.Slice([new Slice(), new Slice(0, 1, 1)]).Relay();
        using var t_count = t.Slice([new Slice(), new Slice(0, 1, 1)]).ToVariable();
        using var round0 = Round.Invoke(y_count)[0];
        using var round1 = Round.Invoke(t_count)[0];
        using var count_loss = MeanSquaredError.Invoke(round0, round1)[0];

        // 2. モンスターの種類（カテゴリカル）の損失
        using var y_type = SliceFunc.Invoke(y, [new Slice(), new Slice(1, 2, 1)])[0]; //y.Data.Value.Slice([new Slice(), new Slice(1, 2, 1)]).ToVariable(y);
        using var t_type = t.Slice([new Slice(), new Slice(1, 2, 1)]).ToVariable();
        using var type_loss = CrossEntropyError.Invoke(y_type, t_type)[0];

        // 3. モンスターのサイズ（連続値）の損失
        using var y_size = SliceFunc.Invoke(y, [new Slice(), new Slice(2, 3, 1)])[0]; //y.Data.Value.Slice([new Slice(), new Slice(2, 3, 1)]).ToVariable(y);
        using var t_size = t.Slice([new Slice(), new Slice(2, 3, 1)]).ToVariable();
        // サイズは相対的な誤差が重要なので、正規化を維持
        using var size_mean = Mean.Invoke(t_size, axis: 0, keepdims: true)[0];
        using var size_std = StdDev.Invoke(t_size, axis: [0], keepdims: true)[0];
        using var y_size_mean = y_size - size_mean;
        using var size_std_eps = size_std + 1e-8f;
        using var normalized_y_size = y_size_mean / size_std_eps;
        using var t_size_mean = t_size - size_mean;
        using var normalized_t_size = t_size_mean / size_std_eps;
        using var size_loss = HuberLoss.Invoke(normalized_y_size, normalized_t_size)[0];

        // 重み付け
        var count_weight = 0.4f;
        var type_weight = 0.3f;
        var size_weight = 0.3f;

        // 最終的な損失を計算
        using var a_loss = count_loss * count_weight;
        using var b_loss = type_loss * type_weight;
        using var c_loss = size_loss * size_weight;
        using var o = a_loss + b_loss;
        using var p = o + c_loss;
        var q = p / batch_size;
        return q;
    }

    public override Variable CalcLoss(Variable y, NDarray t)
    {
        using var scope = this.TrackMemory("CalcLoss");
        try
        {
            var loss = CalculateNormalizedLoss(y, t);

            //// トレーニングフェーズでのみ勾配クリッピングを試みる
            //if (Config.EnableBackprop)
            //{
            //    // 一時的な変数を作成して勾配を計算
            //    using var temp_loss = loss.copy();
            //    temp_loss.Backward(retain_grad: false);

            //    // 勾配が存在する場合のみクリッピングを適用
            //    if (temp_loss.Grad.Value is not null)
            //    {
            //        using var grad_norm = xp.linalg.norm(temp_loss.Grad.Value.Data.Value);
            //        if (grad_norm.asscalar<float>() != float.NaN && grad_norm.asscalar<float>() > GRAD_CLIP_THRESHOLD)
            //        {
            //            var scale = GRAD_CLIP_THRESHOLD / (grad_norm.asscalar<float>() + 1e-8f);
            //            loss = new Variable(loss.Data.Value);
            //            loss.Grad.Value = new Variable(temp_loss.Grad.Value.Data.Value * scale);
            //        }
            //    }
            //}

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
            var batch_size = y.Shape[0];

            // 要素ごとに異なる評価方法を適用
            var y_var = y.Data.Value;
            var t_var = t;

            using var round0 = Round.Invoke(y_var.Slice([new Slice(), new Slice(0, 1, 1)]).ToVariable())[0];
            using var round1 = Round.Invoke(t_var.Slice([new Slice(), new Slice(0, 1, 1)]).ToVariable())[0];
            using var sub_result0 = round0 - round1;
            // モンスターの数（整数値として扱う）
            using var count_diff = Abs.Invoke(sub_result0)[0];

            // モンスターの種類（カテゴリカルな値として扱う）
            using var round2 = Round.Invoke(y_var.Slice([new Slice(), new Slice(1, 2, 1)]).ToVariable())[0];
            using var round3 = Round.Invoke(t_var.Slice([new Slice(), new Slice(1, 2, 1)]).ToVariable())[0];
            using var equal0 = Equal.Invoke(round2, round3)[0];
            using var const0 = Const.Invoke(0f)[0];
            using var const1 = Const.Invoke(1f)[0];
            using var type_diff = Where.Invoke(equal0, const0, const1).Item1[0];

            // サイズ（連続値として扱う）
            using var size0 = y_var.Slice([new Slice(), new Slice(2, 3, 2)]).ToVariable();
            using var size1 = t_var.Slice([new Slice(), new Slice(2, 3, 2)]).ToVariable();
            using var size01 = size0 - size1;
            using var size_diff = Abs.Invoke(size01)[0];

            // 重み付け
            var count_weight = 0.4f;
            var type_weight = 0.3f;
            var size_weight = 0.3f;

            using var mean0 = Mean.Invoke(count_diff)[0];
            using var mean1 = Mean.Invoke(type_diff)[0];
            using var mean2 = Mean.Invoke(size_diff)[0];

            using var mul0 = mean0 * count_weight;
            using var mul1 = mean1 * type_weight;
            using var mul2 = mean2 * size_weight;

            // 重み付き平均を計算
            using var weighted_metric = mul0 + mul1 + mul2;
            
            return weighted_metric.copy();
        }
        finally { }
    }

    protected override Func<NDarray, long> UnitLength => (t) => 1;

    public override ModelType ModelType => ModelType.Regression;
}
