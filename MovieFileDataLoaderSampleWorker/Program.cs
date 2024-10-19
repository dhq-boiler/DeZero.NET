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
workerProcess.SetTrainLoader((ts, batch_size) => new MovieFileDataLoader((MovieFileDataset)ts, () => { (workerProcess.Model as DCNNModel).ResetState(); }));
workerProcess.SetTestLoader((ts, batch_size) => new MovieFileDataLoader((MovieFileDataset)ts, () => { (workerProcess.Model as DCNNModel).ResetState(); }, shuffle: false));
workerProcess.SetModel(() => new DCNNModel());
workerProcess.LoadExistedWeights();
workerProcess.SetOptimizer(model => new SWA(new AdamW().Setup(model), 100, 10, 0.05f));
workerProcess.LoadOptimizer();

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
        return MeanSquaredError.Invoke(y, t.ToVariable())[0];
    }

    public override Variable CalcAccuracy(Variable y, NDarray t)
    {
        return MeanAbsoluteError.Invoke(y, t.ToVariable())[0];
    }

    protected override Func<NDarray, long> UnitLength => (t) => 1;

    public override ModelType ModelType => ModelType.Regression;
}
