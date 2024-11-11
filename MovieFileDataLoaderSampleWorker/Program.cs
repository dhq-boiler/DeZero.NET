using DeZero.NET;
using DeZero.NET.Datasets;
using DeZero.NET.Extensions;
using DeZero.NET.Functions;
using DeZero.NET.Models;
using DeZero.NET.Optimizers;
using MovieFileDataLoaderSampleWorker;

var workerProcess = new WorkerProcess();

workerProcess.SetTrainSet(() => new SampleMovieFileDataset(train: true));
workerProcess.SetTestSet(() => new SampleMovieFileDataset(train: false));
workerProcess.SetTrainLoader((ts, batch_size) => new MovieFileDataLoader((MovieFileDataset)ts, workerProcess.BatchSize, () => { (workerProcess.Model as DCNNModel).ResetState(); }));
workerProcess.SetTestLoader((ts, batch_size) => new MovieFileDataLoader((MovieFileDataset)ts, workerProcess.BatchSize, () => { (workerProcess.Model as DCNNModel).ResetState(); }, shuffle: false));
workerProcess.SetModel(() => new DCNNModel());
workerProcess.LoadExistedWeights();
workerProcess.SetOptimizer(model => new AdamW().Setup(model));
workerProcess.LoadOptimizer();
workerProcess.ResumeState();

workerProcess.Run();

class WorkerProcess : DeZero.NET.Processes.WorkerProcess
{
    public override string PythonDLLPath => @"C:\Users\boiler\AppData\Local\Programs\Python\Python311\python311.dll";
    protected override void InitializeArguments(object[] args)
    {
        Epoch = int.Parse(args[0].ToString());
        BatchSize = int.Parse(args[1].ToString());
        HiddenSize = int.Parse(args[2].ToString());
        EnableGpu = bool.Parse(args[3].ToString());
        RecordFilePath = args[4].ToString().Replace("'", string.Empty);
        DisposeAllInputs = false;
    }

    public override Variable CalcLoss(Variable y, NDarray t)
    {
        // 極端な外れ値を防ぐためのクリッピング
        // 教師データの範囲を基準にクリッピング範囲を決定
        float t_min = t.min().asscalar<float>();
        float t_max = t.max().asscalar<float>();

        // 教師データの範囲より少し広めにクリッピング範囲を設定
        // 予測値が教師データの範囲を完全に外れないように
        float margin = (t_max - t_min) * 0.1f; // マージンとして範囲の10%を追加
        float clip_min = t_min - margin;
        float clip_max = t_max + margin;

        // 予測値をクリッピング
        var clipped_y = Clip.Invoke(y, clip_min, clip_max)[0];

        // クリッピングされた値でMSEを計算
        return MeanSquaredError.Invoke(clipped_y, t.ToVariable())[0];
    }

    public override Variable CalcEvaluationMetric(Variable y, NDarray t)
    {
        return MeanAbsoluteError.Invoke(y, t.ToVariable())[0];
    }

    protected override Func<NDarray, long> UnitLength => (t) => 1;

    public override ModelType ModelType => ModelType.Regression;
}
