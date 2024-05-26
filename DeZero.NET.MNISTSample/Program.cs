
using System.Diagnostics;
using DeZero.NET;
using DeZero.NET.Core;
using DeZero.NET.Datasets;
using DeZero.NET.Functions;
using DeZero.NET.Models;
using DeZero.NET.Optimizers;
using DeZero.NET.Optimizers.HookFunctions;
using DeZero.NET.Exceptions;
using Python.Runtime;

Runtime.PythonDLL = @"C:\Users\boiler\AppData\Local\Programs\Python\Python38\python38.dll";

//Enable GPU
Gpu.Use = true;

var max_epoch = 5;
var batch_size = 100;
var hidden_size = 1000;

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
var model = new MLP([hidden_size, hidden_size, 10], activation: new Relu());
Console.WriteLine("Completed.");

Console.Write($"{DateTime.Now} Start preparing optimizer...");
var optimizer = new Adam().Setup(model);
optimizer.AddHook(new WeightDecay(1e-4f));
Console.WriteLine("Completed.");

Console.WriteLine($"{DateTime.Now} Start training...");

foreach (var epoch in Enumerable.Range(0, max_epoch))
{
    var sum_loss = 0.0;
    var sum_acc = 0.0;
    var count = 0;

    Stopwatch sw = new Stopwatch();

    Console.WriteLine("==================================================================================");
    Console.WriteLine($"epoch : {epoch + 1}");

    sw.Start();

    foreach (var (x, t) in train_loader)
    {
        var y = model.Call(x.ToVariable())[0];
        var softmaxCrossEntropy = new SoftmaxCrossEntropy();
        var loss = softmaxCrossEntropy.Call(Params.New.SetKeywordArg(y, t))[0];
        var accuracy = new Accuracy();
        var acc = accuracy.Call(Params.New.SetKeywordArg(y, t))[0];
        model.ClearGrads();
        loss.Backward();
        optimizer.Update(null);
        sum_loss += loss.Data.Value.asscalar<float>() * t.len;
        sum_acc += acc.Data.Value.asscalar<float>() * t.len;
        count++;
    }

    Console.WriteLine($"train loss: {sum_loss / train_set.Length}, accuracy: {sum_acc / train_set.Length}");
    
    var test_loss = 0.0;
    var test_acc = 0.0;
    using (var config = ConfigExtensions.NoGrad())
    {
        foreach (var (x, t) in test_loader)
        {
            var y = model.Call(x.ToVariable())[0];
            var softmaxCrossEntropy = new SoftmaxCrossEntropy();
            var loss = softmaxCrossEntropy.Call(Params.New.SetKeywordArg(y, t))[0];
            var accuracy = new Accuracy();
            var acc = accuracy.Call(Params.New.SetKeywordArg(y, t))[0];
            test_loss += loss.Data.Value.asscalar<float>() * t.len;
            test_acc += acc.Data.Value.asscalar<float>() * t.len;
        }
    }

    sw.Stop();
    
    Console.WriteLine($"test loss: {test_loss / test_set.Length}, test acc: {test_acc / test_set.Length}");
    Console.WriteLine($"time : {(int)(sw.ElapsedMilliseconds / 1000 / 60)}m{(sw.ElapsedMilliseconds / 1000 % 60)}s");
}

Console.WriteLine("==================================================================================");
Console.WriteLine($"{DateTime.Now} Finish training.");