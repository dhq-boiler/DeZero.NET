using ClosedXML.Excel;
using DeZero.NET.Core;
using DeZero.NET.Functions;
using DeZero.NET.Optimizers;
using DeZero.NET.Recorder;
using Python.Runtime;
using System.Diagnostics;
using System.Text;

namespace DeZero.NET.Processes
{
    public abstract class WorkerProcess
    {
        public object[] Args { get; }
        public int Epoch { get; set; }
        public int BatchSize { get; set; }
        public int HiddenSize { get; set; }
        public bool EnableGpu { get; set; }
        public string RecordFilePath { get; set; }

        public DeZero.NET.Datasets.Dataset TrainSet { get; private set; }
        public DeZero.NET.Datasets.Dataset TestSet { get; private set; }
        public DeZero.NET.Datasets.DataLoader TrainLoader { get; private set; }
        public DeZero.NET.Datasets.DataLoader TestLoader { get; private set; }
        public Models.Model Model { get; private set; }
        public Optimizer Optimizer { get; private set; }

        public WorkerProcess()
        {
            AppDomain.CurrentDomain.ProcessExit += (sender, e) =>
            {
                Environment.Exit(-1);
            };

            Console.OutputEncoding = Encoding.UTF8;
            Args = Environment.GetCommandLineArgs().Skip(1).ToArray();
            InitializeArguments(Args);
            InitializePython();
            InitializeXp();
            SetGpuUse();
        }

        private static void InitializeXp()
        {
            Console.Write($"{DateTime.Now} xp.Initialize...");
            xp.Initialize();
            Console.WriteLine("Completed.");
        }

        public void SetTrainSet(Func<DeZero.NET.Datasets.Dataset> trainSet)
        {
            Console.Write($"{DateTime.Now} Start preparing train_set...");
            TrainSet = trainSet();
            Console.WriteLine("Completed.");
        }

        public void SetTestSet(Func<DeZero.NET.Datasets.Dataset> testSet)
        {
            Console.Write($"{DateTime.Now} Start preparing test_set...");
            TestSet = testSet();
            Console.WriteLine("Completed.");
        }

        public void SetTrainLoader(Func<DeZero.NET.Datasets.Dataset, int, DeZero.NET.Datasets.DataLoader> trainLoader)
        {
            Console.Write($"{DateTime.Now} Start preparing train_loader...");
            TrainLoader = trainLoader(this.TrainSet, BatchSize);
            Console.WriteLine("Completed.");
        }

        public void SetTestLoader(Func<DeZero.NET.Datasets.Dataset, int, DeZero.NET.Datasets.DataLoader> testLoader)
        {
            Console.Write($"{DateTime.Now} Start preparing test_loader...");
            TestLoader = testLoader(this.TestSet, BatchSize);
            Console.WriteLine("Completed.");
        }

        public void SetModel(Func<Models.Model> model)
        {
            Console.Write($"{DateTime.Now} Start preparing model...");
            Model = model();
            Console.WriteLine("Completed.");
        }

        public virtual void LoadExistedWeights()
        {
            Directory.CreateDirectory("weights");
            if (Directory.EnumerateFiles("weights").Any())
            {
                Console.Write($"{DateTime.Now} Start loading weights...");
                Model.LoadWeights();
                Console.WriteLine("Completed.");
            }
        }

        public void SetOptimizer(Func<Models.Model, Optimizer> optimizer)
        {
            Console.Write($"{DateTime.Now} Start preparing optimizer...");
            Optimizer = optimizer(Model);
            Console.WriteLine("Completed.");
        }

        public abstract string PythonDLLPath { get; }

        protected abstract void InitializeArguments(object[] args);

        public void Run()
        {
            var sum_loss = 0.0;
            var sum_acc = 0.0;
            var count = 0;

            Stopwatch sw = new Stopwatch();

            Console.WriteLine($"{DateTime.Now} Start training...");
            Console.WriteLine("==================================================================================");
            Console.WriteLine($"epoch : {Epoch}");

            sw.Start();

            foreach (var (x, t) in TrainLoader)
            {
                using var y = Model.Call(x.ToVariable())[0];
                var loss = CalcLoss(y, t);
                var accuracy = new Accuracy();
                using var acc = accuracy.Call(Params.New.SetKeywordArg(y, t))[0];
                using var total_loss = CalcAdditionalLoss(loss);
                Model.ClearGrads();
                total_loss.Backward(retain_grad: false);
                Model.DisposeAllInputs();
                Optimizer.Update(null);
                sum_loss += total_loss.Data.Value.asscalar<float>() * t.len;
                sum_acc += acc.Data.Value.asscalar<float>() * t.len;
                count++;
                x.Dispose();
                t.Dispose();
                loss.Dispose();
                GC.Collect();
                Finalizer.Instance.Collect();
            }

            Console.WriteLine($"train loss: {sum_loss / TrainSet.Length}, accuracy: {sum_acc / TrainSet.Length}");

            var test_loss = 0.0;
            var test_acc = 0.0;
            using (var config = ConfigExtensions.NoGrad())
            {
                foreach (var (x, t) in TestLoader)
                {
                    using var y = Model.Call(x.ToVariable())[0];
                    Model.DisposeAllInputs();
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

            Console.WriteLine($"test loss: {test_loss / TestSet.Length}, test acc: {test_acc / TestSet.Length}");
            Console.WriteLine($"time : {(int)(sw.ElapsedMilliseconds / 1000 / 60)}m{(sw.ElapsedMilliseconds / 1000 % 60)}s");
            Console.WriteLine("==================================================================================");

            EpochResult epochResult = new EpochResult
            {
                Epoch = Epoch,
                TrainLoss = sum_loss / TrainSet.Length,
                TrainAccuracy = sum_acc / TrainSet.Length,
                TestLoss = test_loss / TestSet.Length,
                TestAccuracy = test_acc / TestSet.Length,
                ElapsedMilliseconds = sw.ElapsedMilliseconds
            };
            WriteResultToRecordFile(epochResult);
            SaveWeights();
            ExitSequence();
        }

        private void ExitSequence()
        {
            //親プロセスに終了を通知
            File.Create("signal").Close();

            //親プロセスからKillされるか、自害するか
            Environment.Exit(0);
        }

        private void SaveWeights()
        {
            Console.Write($"{DateTime.Now} Save weights...");
            Model.SaveWeights();
            Console.WriteLine("Completed.");
        }

        private void WriteResultToRecordFile(EpochResult epochResult)
        {
            Console.Write($"{DateTime.Now} Save XLSX:{RecordFilePath} ...");
            using var workbook = File.Exists(RecordFilePath) ? new XLWorkbook(RecordFilePath) : new XLWorkbook();
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

            if (File.Exists(RecordFilePath))
            {
                workbook.Save();
            }
            else
            {
                workbook.SaveAs(RecordFilePath);
            }
            Console.WriteLine("Completed.");
        }

        protected virtual Variable CalcAdditionalLoss(Variable loss)
        {
            return loss + L2Regularization.Invoke(Model.Params(), new NDarray(0.01).ToVariable())[0];
        }

        protected virtual Variable CalcLoss(Variable y, NDarray t)
        {
            return SoftmaxCrossEntropy.Invoke(y, t.ToVariable())[0];
        }

        private void InitializePython()
        { 
            Runtime.PythonDLL = PythonDLLPath;
            PythonEngine.Initialize();
        }

        private void SetGpuUse()
        {
            Gpu.Use = EnableGpu;
            Console.WriteLine($"{DateTime.Now} {(Gpu.Available && Gpu.Use ? "GPU Enabled" : "GPU Disabled")}");
        }
    }
}
