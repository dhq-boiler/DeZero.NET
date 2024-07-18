using DeZero.NET.Datasets;
using DeZero.NET.Optimizers;
using Dtype = DeZero.NET.Dtype;
using L = DeZero.NET.Layers;

var workerProcess = new WorkerProcess();

workerProcess.SetTrainSet(() => new MNIST(train: true));
workerProcess.SetTestSet(() => new MNIST(train: false));
workerProcess.SetTrainLoader((ts, batch_size) => new DataLoader(ts, batch_size));
workerProcess.SetTestLoader((ts, batch_size) => new DataLoader(ts, batch_size, shuffle: false));
workerProcess.SetModel(() => new DeZero.NET.Models.Sequential([
        new L.Convolution.Conv2d(64, 7, Dtype.float32),
        new L.Normalization.BatchNorm(),
        new L.Activation.ReLU(),
        new L.Convolution.MaxPooling((2, 2), (2, 2), (0, 0)),
        new L.Convolution.Conv2d(128, 7, Dtype.float32),
        new L.Normalization.BatchNorm(),
        new L.Activation.ReLU(),
        new L.Convolution.MaxPooling((2, 2), (2, 2), (0, 0)),
        new L.Convolution.Conv2d(256, 7, Dtype.float32, pad:3),
        new L.Normalization.BatchNorm(),
        new L.Activation.ReLU(),
        new L.Linear.Flatten(),
        new L.Linear.Linear(256),
        new L.Normalization.BatchNorm(),
        new L.Activation.ReLU(),
        new L.Linear.Dropout(0.25),
        new L.Linear.Linear(10),
        new L.Activation.Softmax()
    ]));
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
    }
}
