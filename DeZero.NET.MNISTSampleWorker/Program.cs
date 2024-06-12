
using ClosedXML.Excel;
using DeZero.NET;
using DeZero.NET.Core;
using DeZero.NET.Datasets;
using DeZero.NET.Functions;
using DeZero.NET.Optimizers;
using DeZero.NET.Optimizers.HookFunctions;
using DeZero.NET.Recorder;
using Python.Runtime;
using System.Diagnostics;
using Dtype = DeZero.NET.Dtype;
using L = DeZero.NET.Layers;

AppDomain.CurrentDomain.ProcessExit += (sender, e) =>
{
    Environment.Exit(-1);
};

var epoch = int.Parse(Environment.GetCommandLineArgs()[1]);
var batch_size = int.Parse(Environment.GetCommandLineArgs()[2]);
var hidden_size = int.Parse(Environment.GetCommandLineArgs()[3]);
var enableGpu = bool.Parse(Environment.GetCommandLineArgs()[4]);

Runtime.PythonDLL = @"C:\Users\boiler\AppData\Local\Programs\Python\Python311\python311.dll";
PythonEngine.Initialize();

const string xlsx_path = "MNIST_result.xlsx";

//Enable GPU
Gpu.Use = enableGpu;
Console.WriteLine($"{DateTime.Now} {(Gpu.Available && Gpu.Use ? "GPU Enabled" : "GPU Disabled")}");

Console.Write($"{DateTime.Now} xp.Initialize...");
xp.Initialize();
Console.WriteLine("Completed.");

Console.Write($"{DateTime.Now} Start preparing train_set...");
var train_set = new MNIST(train: true);
Console.WriteLine("Completed.");

Console.Write($"{DateTime.Now} Start preparing test_set...");
var test_set = new MNIST(train: false);
Console.WriteLine("Completed.");

Console.Write($"{DateTime.Now} Start preparing train_loader...");
var train_loader = new DataLoader(train_set, batch_size);
Console.WriteLine("Completed.");

Console.Write($"{DateTime.Now} Start preparing test_loader...");
var test_loader = new DataLoader(test_set, batch_size, shuffle: false);
Console.WriteLine("Completed.");

Console.Write($"{DateTime.Now} Start preparing model...");
//var model = new MLP([hidden_size, hidden_size, 10], activation: new DeZero.NET.Functions.ReLU());
//var model = new DeZero.NET.Models.Sequential([
//    new L.Standardization.WeightStandardization(new L.Convolution.Conv2d(32, 3, Dtype.float32)),
//    new L.Activation.ReLU(),
//    new L.Standardization.WeightStandardization(new L.Convolution.Conv2d(64, 3, Dtype.float32)),
//    new L.Activation.ReLU(),
//    new L.Convolution.MaxPooling((2, 2), (1, 1), (0, 0)),
//    new L.Linear.Dropout(0.25),
//    new L.Linear.Flatten(),
//    new L.Normalization.WeightNorm(new L.Linear.Linear(128)),
//    new L.Activation.ReLU(),
//    new L.Linear.Dropout(0.5),
//    new L.Normalization.WeightNorm(new L.Linear.Linear(10)),
//    new L.Activation.Softmax()
//]);
//var model = new DeZero.NET.Models.Sequential([
//    new L.Convolution.Conv2d(32, 3, Dtype.float32),
//    new L.Activation.ReLU(),
//    new L.Convolution.Conv2d(64, 3, Dtype.float32),
//    new L.Activation.ReLU(),
//    new L.Convolution.MaxPooling((2, 2), (1, 1), (0, 0)),
//    new L.Linear.Dropout(0.25),
//    new L.Linear.Flatten(),
//    new L.Linear.Linear(128),
//    new L.Activation.ReLU(),
//    new L.Linear.Dropout(0.5),
//    new L.Linear.Linear(10),
//    new L.Activation.Softmax()
//]);
//var model = new DeZero.NET.Models.Sequential([
//        new L.Convolution.Conv2d(32, 3, Dtype.float32),
//        new L.Activation.ReLU(),
//        new L.Convolution.MaxPooling((2, 2), (1, 1), (0, 0)),
//        new L.Convolution.Conv2d(64, 3, Dtype.float32),
//        new L.Activation.ReLU(),
//        new L.Convolution.MaxPooling((2, 2), (1, 1), (0, 0)),
//        new L.Convolution.Conv2d(64, 3, Dtype.float32),
//        new L.Activation.ReLU(),
//        new L.Linear.Flatten(),
//        new L.Linear.Linear(64),
//        new L.Activation.ReLU(),
//        new L.Linear.Linear(10),
//        new L.Activation.Softmax()
//    ]
//);
var model = new DeZero.NET.Models.Sequential([
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
    ]
);
Console.WriteLine("Completed.");
Directory.CreateDirectory("weights");
if (Directory.EnumerateFiles("weights").Any())
{
    Console.Write($"{DateTime.Now} Start loading weights...");
    model.LoadWeights();
    Console.WriteLine("Completed.");
}

Console.Write($"{DateTime.Now} Start preparing optimizer...");
var optimizer = new Adam().Setup(model);
optimizer.AddHook(new WeightDecay(1e-4f));
Console.WriteLine("Completed.");

var sum_loss = 0.0;
var sum_acc = 0.0;
var count = 0;

Stopwatch sw = new Stopwatch();

Console.WriteLine($"{DateTime.Now} Start training...");

Console.WriteLine("==================================================================================");
Console.WriteLine($"epoch : {epoch}");

sw.Start();

foreach (var (x, t) in train_loader)
{
    using var y = model.Call(x.ToVariable())[0];
    var loss = SoftmaxCrossEntropy.Invoke(y, t.ToVariable())[0];
    var accuracy = new Accuracy();
    using var acc = accuracy.Call(Params.New.SetKeywordArg(y, t))[0];
    using var total_loss = loss + L2Regularization.Invoke(model.Params(), new NDarray(0.01).ToVariable())[0];
    model.ClearGrads();
    total_loss.Backward(retain_grad: false);
    model.DisposeAllInputs();
    optimizer.Update(null);
    sum_loss += total_loss.Data.Value.asscalar<float>() * t.len;
    sum_acc += acc.Data.Value.asscalar<float>() * t.len;
    count++;
    x.Dispose();
    t.Dispose();
    loss.Dispose();
    GC.Collect();
    Finalizer.Instance.Collect();
}

Console.WriteLine($"train loss: {sum_loss / train_set.Length}, accuracy: {sum_acc / train_set.Length}");

var test_loss = 0.0;
var test_acc = 0.0;
using (var config = ConfigExtensions.NoGrad())
{
    foreach (var (x, t) in test_loader)
    {
        using var y = model.Call(x.ToVariable())[0];
        model.DisposeAllInputs();
        var softmaxCrossEntropy = new SoftmaxCrossEntropy();
        using var loss = softmaxCrossEntropy.Call(Params.New.SetKeywordArg(y, t))[0];
        var accuracy = new Accuracy();
        using var acc = accuracy.Call(Params.New.SetKeywordArg(y, t))[0];
        test_loss += loss.Data.Value.asscalar<float>() * t.len;
        test_acc += acc.Data.Value.asscalar<float>() * t.len;
        x.Dispose();
        t.Dispose();
        GC.Collect();
        Finalizer.Instance.Collect();
    }
}

sw.Stop();

Console.WriteLine($"test loss: {test_loss / test_set.Length}, test acc: {test_acc / test_set.Length}");
Console.WriteLine($"time : {(int)(sw.ElapsedMilliseconds / 1000 / 60)}m{(sw.ElapsedMilliseconds / 1000 % 60)}s");
Console.WriteLine("==================================================================================");

EpochResult epochResult = new EpochResult
{
    Epoch = epoch,
    TrainLoss = sum_loss / train_set.Length,
    TrainAccuracy = sum_acc / train_set.Length,
    TestLoss = test_loss / test_set.Length,
    TestAccuracy = test_acc / test_set.Length,
    ElapsedMilliseconds = sw.ElapsedMilliseconds
};

Console.Write($"{DateTime.Now} Save XLSX:{xlsx_path} ...");
using var workbook = File.Exists(xlsx_path) ? new XLWorkbook(xlsx_path) : new XLWorkbook();
var worksheet = workbook.Worksheets.SingleOrDefault(s => s.Name == "data") ?? workbook.AddWorksheet("data");
worksheet.Cell(1, 1).Value = "epoch";
worksheet.Cell(2, 1).Value = "train_loss";
worksheet.Cell(3, 1).Value = "train_accuracy";
worksheet.Cell(4, 1).Value = "test_loss";
worksheet.Cell(5, 1).Value = "test_accuracy";
worksheet.Cell(6, 1).Value = "h";
worksheet.Cell(7, 1).Value = "m";
worksheet.Cell(8, 1).Value = "s";

worksheet.Cell(1, epochResult.Epoch + 1).Value = epochResult.Epoch;
worksheet.Cell(2, epochResult.Epoch + 1).Value = epochResult.TrainLoss;
worksheet.Cell(3, epochResult.Epoch + 1).Value = epochResult.TrainAccuracy;
worksheet.Cell(4, epochResult.Epoch + 1).Value = epochResult.TestLoss;
worksheet.Cell(5, epochResult.Epoch + 1).Value = epochResult.TestAccuracy;
worksheet.Cell(6, epochResult.Epoch + 1).Value = (int)(epochResult.ElapsedMilliseconds / 1000 / 60 / 60);
worksheet.Cell(7, epochResult.Epoch + 1).Value = (int)(epochResult.ElapsedMilliseconds / 1000 / 60 % 60);
worksheet.Cell(8, epochResult.Epoch + 1).Value = (int)(epochResult.ElapsedMilliseconds / 1000 % 60 % 60);

if (File.Exists(xlsx_path))
{
    workbook.Save();
}
else
{
    workbook.SaveAs(xlsx_path);
}
Console.WriteLine("Completed.");

Console.Write($"{DateTime.Now} Save weights...");
model.SaveWeights();
Console.WriteLine("Completed.");

//親プロセスに終了を通知
File.Create("signal").Close();

//親プロセスからKillされるか、自害するか
Environment.Exit(0);
