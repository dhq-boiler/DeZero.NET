using ClosedXML;
using ClosedXML.Excel;
using DeZero.NET.Core;
using DeZero.NET.Datasets;
using DeZero.NET.Extensions;
using DeZero.NET.Functions;
using DeZero.NET.Models;
using DeZero.NET.Optimizers;
using DeZero.NET.Recorder;
using Python.Runtime;
using System.Diagnostics;
using System.Text;
using static DeZero.NET.Recorder.EpochResult;

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
        public bool DisposeAllInputs { get; set; } = false;

        public DeZero.NET.Datasets.Dataset TrainSet { get; private set; }
        public DeZero.NET.Datasets.Dataset TestSet { get; private set; }
        public DeZero.NET.Datasets.IDataProvider TrainLoader { get; private set; }
        public DeZero.NET.Datasets.IDataProvider TestLoader { get; private set; }
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

        public abstract ModelType ModelType { get; }

        private static void InitializeXp()
        {
            Console.Write($"{DateTime.Now} xp.Initialize...");
            xp.Initialize();
            Console.WriteLine("Completed.");
        }

        /// <summary>
        /// トレーニングデータセットを設定します.
        /// </summary>
        /// <param name="trainSet">トレーニングデータセット</param>
        public void SetTrainSet(Func<DeZero.NET.Datasets.Dataset> trainSet)
        {
            Console.Write($"{DateTime.Now} Start preparing train_set...");
            TrainSet = trainSet();
            Console.WriteLine("Completed.");
        }

        /// <summary>
        /// テストデータセットを設定します.
        /// </summary>
        /// <param name="testSet">テストデータセット</param>
        public void SetTestSet(Func<DeZero.NET.Datasets.Dataset> testSet)
        {
            Console.Write($"{DateTime.Now} Start preparing test_set...");
            TestSet = testSet();
            Console.WriteLine("Completed.");
        }

        /// <summary>
        /// トレーニングデータローダーを設定します.
        /// </summary>
        /// <param name="trainLoader">トレーニングデータローダー</param>
        public void SetTrainLoader(Func<DeZero.NET.Datasets.Dataset, int, DeZero.NET.Datasets.IDataProvider> trainLoader)
        {
            Console.Write($"{DateTime.Now} Start preparing train_loader...");
            TrainLoader = trainLoader(this.TrainSet, BatchSize);
            TrainLoader.OnSwitchDataFile += (sum_loss, sum_err, sum_acc, movie_file_path, sw) =>
            {
                EpochResult epochResult = new EpochResult
                {
                    ModelType = ModelType,
                    Epoch = Epoch,
                    TargetDataFile = movie_file_path,
                    TrainOrTestType = EpochResult.TrainOrTest.Train,
                    TrainLoss = sum_loss / TrainLoader.Length,
                    TrainError = sum_err / TrainLoader.Length,
                    TrainAccuracy = sum_acc / TrainLoader.Length,
                    ElapsedMilliseconds = sw.ElapsedMilliseconds
                };
                WriteResultToRecordFile(epochResult);
            };
            Console.WriteLine("Completed.");
        }

        /// <summary>
        /// テストデータローダーを設定します.
        /// </summary>
        /// <param name="testLoader">テストデータローダー</param>
        public void SetTestLoader(Func<DeZero.NET.Datasets.Dataset, int, DeZero.NET.Datasets.IDataProvider> testLoader)
        {
            Console.Write($"{DateTime.Now} Start preparing test_loader...");
            TestLoader = testLoader(this.TestSet, BatchSize);
            TestLoader.OnSwitchDataFile += (sum_loss, sum_err, sum_acc, movie_file_path, sw) =>
            {
                var epochResult = new EpochResult
                {
                    ModelType = ModelType,
                    Epoch = Epoch,
                    TargetDataFile = movie_file_path,
                    TrainOrTestType = EpochResult.TrainOrTest.Test,
                    TestLoss = sum_loss / TestLoader.Length,
                    TestError = sum_err / TestLoader.Length,
                    TestAccuracy = sum_acc / TestLoader.Length,
                    ElapsedMilliseconds = sw.ElapsedMilliseconds
                };
                WriteResultToRecordFile(epochResult);
            };
            Console.WriteLine("Completed.");
        }

        /// <summary>
        /// モデルを設定します.
        /// </summary>
        /// <param name="model">モデル</param>
        public void SetModel(Func<Models.Model> model)
        {
            Console.Write($"{DateTime.Now} Start preparing model...");
            Model = model();
            Console.WriteLine("Completed.");
        }

        /// <summary>
        /// 既存の重みを読み込みます.
        /// </summary>
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

        /// <summary>
        /// 重みを保存します.
        /// </summary>
        public void SaveWeights()
        {
            Console.Write($"{DateTime.Now} Save weights...");
            Model.SaveWeights();
            Console.WriteLine("Completed.");
        }

        /// <summary>
        /// オプティマイザを設定します.
        /// </summary>
        /// <param name="optimizer">オプティマイザ</param>
        public void SetOptimizer(Func<Models.Model, Optimizer> optimizer)
        {
            Console.Write($"{DateTime.Now} Start preparing optimizer...");
            Optimizer = optimizer(Model);
            Console.WriteLine("Completed.");
        }

        /// <summary>
        /// オプティマイザの状態を読み込みます.
        /// </summary>
        public void LoadOptimizer()
        {
            Console.Write($"{DateTime.Now} Start optimizer states...");
            Directory.CreateDirectory("optimizer");
            if (Directory.EnumerateFiles("optimizer").Any())
            {
                Optimizer.LoadParameters();
            }
            Console.WriteLine("Completed.");
        }

        /// <summary>
        /// オプティマイザの状態を保存します.
        /// </summary>
        public void SaveOptimizer()
        {
            Console.Write($"{DateTime.Now} Save optimizer states...");
            Optimizer.SaveParameters();
            Console.WriteLine("Completed.");
        }

        public void ResumeState()
        {
            if (!File.Exists(RecordFilePath))
            {
                return;
            }

            if (TrainSet is not MovieFileDataset || TestSet is not MovieFileDataset)
            {
                throw new InvalidOperationException();
            }

            //以前の起動状態をレジュームする旨をコンソール出力する
            Console.Write($"{DateTime.Now} Resume state...");

            // ワークブックをRecordFilePathから読み込みます
            using var workbook = new XLWorkbook(RecordFilePath);
            var worksheet = workbook.Worksheet(1);

            WriteTemplate(Epoch, worksheet, (TrainSet as MovieFileDataset).MovieFilePaths.Count(), (TestSet as MovieFileDataset).MovieFilePaths.Count());

            if (File.Exists(RecordFilePath))
            {
                workbook.Save();
            }
            else
            {
                workbook.SaveAs(RecordFilePath);
            }

            // 各列を取得します
            var epochColumn = worksheet.Column(2);
            var trainOrTestColumn = worksheet.Column(3);
            var movieFileColumn = worksheet.Column(4);
            var lossColumn = worksheet.Column(5);

            // 現在のエポックに対応する行番号を取得します
            var currentEpochRowNumbers = epochColumn.CellsUsed()
                                                    .Where(cell => int.TryParse(cell.GetValue<string>(), out var value) ? value == Epoch : false)
                                                    .Select(c => c.Address.RowNumber)
                                                    .OrderBy(x => x)
                                                    .ToArray();

            // "train"行をフィルタリングします
            var trainRows = currentEpochRowNumbers.Take((TrainSet as MovieFileDataset).MovieFilePaths.Count()).ToArray();

            // "train"行のmovieファイル名を取得します
            var trainRows_movieFiles = trainRows.Select(r => movieFileColumn.Cell(r).GetString()).ToArray();

            // TrainLoaderのMovieIndexを設定します
            TrainLoader.MovieIndex = xp.array(trainRows_movieFiles.ToList().Select(x => (TrainSet as MovieFileDataset).MovieFilePaths.Select((v, index) => new { Value = v, Index = index })
                                                                                                                                     .First(y => y.Value.Equals(x)).Index)
                                                                                                                                     .ToArray());

            // "train"行のloss値を取得します
            var lossTrainRows = trainRows.Select(r => lossColumn.Cell(r).GetValue<float?>()).Where(x => x is not null).ToArray();

            // TrainLoaderのCurrentMovieIndexを設定します
            TrainLoader.CurrentMovieIndex = lossTrainRows.Length;

            // "test"行をフィルタリングします
            var testRows = currentEpochRowNumbers.Skip((TrainSet as MovieFileDataset).MovieFilePaths.Count() + 1).Take((TestSet as MovieFileDataset).MovieFilePaths.Count()).ToArray();

            // "test"行のmovieファイル名を取得します
            var testRows_movieFiles = testRows.Select(r => movieFileColumn.Cell(r).GetString()).ToArray();

            var testArr = testRows_movieFiles.ToList().Select(x => (TestSet as MovieFileDataset).MovieFilePaths.Select((v, index) => new { Value = v, Index = index })
                                                                                                                                  .First(y => y.Value.Equals(x)).Index)
                                                                                                                                  .ToArray();
            // TestLoaderのMovieIndexを設定します
            TestLoader.MovieIndex = xp.array(testArr);

            // "test"行のloss値を取得します
            var lossTestRows = testRows.Select(r => lossColumn.Cell(r).GetValue<float?>()).Where(x => x is not null).ToArray();

            // TestLoaderのCurrentMovieIndexを設定します
            TestLoader.CurrentMovieIndex = lossTestRows.Length;

            Console.WriteLine("Completed.");
        }

        /// <summary>
        /// python311.dll のパスを取得します.
        /// </summary>
        /// <remarks>サブクラスで実装します.</remarks>
        public abstract string PythonDLLPath { get; }

        /// <summary>
        /// 親プロセスから渡された引数を初期化します.
        /// </summary>
        /// <param name="args">親プロセスから渡された引数の配列</param>
        protected abstract void InitializeArguments(object[] args);

        protected virtual Func<NDarray, long> UnitLength => (t) => t.len;

        /// <summary>
        /// トレーニングとテストを実行します.
        /// </summary>
        public void Run()
        {
            var sum_loss = 0.0;
            var sum_err = 0.0;
            var sum_acc = 0.0;
            var count = 0;

            Stopwatch sw = new Stopwatch();

            Console.WriteLine($"{DateTime.Now} Start training...");
            Console.WriteLine("==================================================================================");
            Console.WriteLine($"epoch : {Epoch}");
            Console.WriteLine($"training...");

            sw.Start();

            foreach (var (x, t) in TrainLoader)
            {
                using var y = Model.Call(x.ToVariable())[0];
                using var loss = CalcLoss(y, t);
                using var acc = CalcAccuracy(y, t);
                using var total_loss = CalcAdditionalLoss(loss);
                Model.ClearGrads();
                total_loss.Backward(retain_grad: false);
                if (DisposeAllInputs)
                {
                    Model.DisposeAllInputs();
                }
                Optimizer.Update(null);
                sum_loss += total_loss.Data.Value.asscalar<float>() * UnitLength(t);
                switch (ModelType)
                {
                    case ModelType.Regression:
                        sum_err += acc.Data.Value.asscalar<float>() * UnitLength(t);
                        break;
                    case ModelType.Classification:
                        sum_acc += acc.Data.Value.asscalar<float>() * UnitLength(t);
                        break;
                }
                TrainLoader.NotifyEvalValues(sum_loss, sum_err, sum_acc, sw);
                count++;
                GC.Collect();
                Finalizer.Instance.Collect();
            }

            EpochResult epochResult = new EpochResult
            {
                ModelType = ModelType,
                Epoch = Epoch,
                TrainOrTestType = EpochResult.TrainOrTest.TrainTotal,
                TrainLoss = sum_loss / TrainLoader.Length,
                TrainError = sum_err / TrainLoader.Length,
                TrainAccuracy = sum_acc / TrainLoader.Length,
                ElapsedMilliseconds = sw.ElapsedMilliseconds
            };

            if (double.IsNaN(epochResult.TrainLoss))
            {
                Console.WriteLine("skip.");
            }
            else
            {
                switch (ModelType)
                {
                    case ModelType.Regression:
                        Console.WriteLine($"train loss: {sum_loss / TrainLoader.Length}, error: {sum_err / TrainLoader.Length}");
                        break;
                    case ModelType.Classification:
                        Console.WriteLine($"train loss: {sum_loss / TrainLoader.Length}, accuracy: {sum_acc / TrainLoader.Length}");
                        break;
                }
                WriteResultToRecordFile(epochResult);
            }

            Console.WriteLine();
            Console.WriteLine($"testing...");

            var test_loss = 0.0;
            var test_err = 0.0;
            var test_acc = 0.0;
            using (var config = ConfigExtensions.NoGrad())
            {
                foreach (var (x, t) in TestLoader)
                {
                    using var y = Model.Call(x.ToVariable())[0];
                    if (DisposeAllInputs)
                    {
                        Model.DisposeAllInputs();
                    }
                    using var loss = CalcLoss(y, t);
                    using var acc = CalcAccuracy(y, t);
                    test_loss += loss.Data.Value.asscalar<float>() * UnitLength(t);
                    switch (ModelType)
                    {
                        case ModelType.Regression:
                            test_err += acc.Data.Value.asscalar<float>() * UnitLength(t);
                            break;
                        case ModelType.Classification:
                            test_acc += acc.Data.Value.asscalar<float>() * UnitLength(t);
                            break;
                    }
                    TestLoader.NotifyEvalValues(test_loss, test_err, test_acc, sw);
                    GC.Collect();
                    Finalizer.Instance.Collect();
                }
            }

            sw.Stop();

            switch (ModelType)
            {
                case ModelType.Regression:
                    Console.WriteLine($"test loss: {test_loss / TestLoader.Length}, test error: {test_err / TestLoader.Length}");
                    break;
                case ModelType.Classification:
                    Console.WriteLine($"test loss: {test_loss / TestLoader.Length}, test acc: {test_acc / TestLoader.Length}");
                    break;
            }
            Console.WriteLine($"time : {(int)(sw.ElapsedMilliseconds / 1000 / 60)}m{(sw.ElapsedMilliseconds / 1000 % 60)}s");
            Console.WriteLine("==================================================================================");

            epochResult = new EpochResult
            {
                ModelType = ModelType,
                Epoch = Epoch,
                TrainOrTestType = EpochResult.TrainOrTest.TestTotal,
                TestLoss = test_loss / TestLoader.Length,
                TestError = test_err / TestLoader.Length,
                TestAccuracy = test_acc / TestLoader.Length,
                ElapsedMilliseconds = sw.ElapsedMilliseconds
            };
            WriteResultToRecordFile(epochResult);
            SaveWeights();
            SaveOptimizer();
            ExitSequence();
        }

        private void ExitSequence()
        {
            //親プロセスに終了を通知
            File.Create("signal").Close();

            //親プロセスからKillされるか,自害するか
            Environment.Exit(0);
        }

        private void WriteResultToRecordFile(EpochResult epochResult)
        {
            Console.Write($"{DateTime.Now} Save XLSX:{RecordFilePath} ...");
            if (this.TrainLoader is MovieFileDataLoader && TestLoader is MovieFileDataLoader)
            {
                WriteVerticalResult(epochResult);
            }
            else
            {
                WriteHorizontalResult(epochResult);
            }
            Console.WriteLine("Completed.");
        }

        private void WriteVerticalResult(EpochResult epochResult)
        {
            using var workbook = File.Exists(RecordFilePath) ? new XLWorkbook(RecordFilePath) : new XLWorkbook();
            var worksheet = workbook.Worksheets.SingleOrDefault(s => s.Name == "data") ?? workbook.AddWorksheet("data");
            worksheet.Cell(1, 1).Value = "No";
            worksheet.Cell(1, 2).Value = "epoch";
            worksheet.Cell(1, 3).Value = "train or test";
            worksheet.Cell(1, 4).Value = "movie file";
            worksheet.Cell(1, 5).Value = "loss";
            worksheet.Cell(1, 6).Value = "error";
            worksheet.Cell(1, 7).Value = "h";
            worksheet.Cell(1, 8).Value = "m";
            worksheet.Cell(1, 9).Value = "s";

            WriteTemplate(epochResult.Epoch, worksheet, (TrainSet as MovieFileDataset).MovieFilePaths.Count(), (TestSet as MovieFileDataset).MovieFilePaths.Count());

            var nextNo = GetNextNo(worksheet);
            var currentRow = nextNo + 1;
            worksheet.Cell(currentRow, 1).Value = nextNo;
            worksheet.Cell(currentRow, 2).Value = epochResult.Epoch;
            worksheet.Cell(currentRow, 3).Value = epochResult.TrainOrTestType.ToString().ToLower();
            worksheet.Cell(currentRow, 4).Value = epochResult.TargetDataFile;
            var col5Value = epochResult.TrainOrTestType switch
            {
                EpochResult.TrainOrTest.Train => epochResult.TrainLoss,
                EpochResult.TrainOrTest.TrainTotal => epochResult.TrainLoss,
                EpochResult.TrainOrTest.Test => epochResult.TestLoss,
                EpochResult.TrainOrTest.TestTotal => epochResult.TestLoss,
                _ => 0
            };
            worksheet.Cell(currentRow, 5).Value = double.IsNaN(col5Value) ? string.Empty : col5Value;
            var col6Value = epochResult.TrainOrTestType switch
            {
                EpochResult.TrainOrTest.Train => epochResult.TrainError,
                EpochResult.TrainOrTest.TrainTotal => epochResult.TrainError,
                EpochResult.TrainOrTest.Test => epochResult.TestError,
                EpochResult.TrainOrTest.TestTotal => epochResult.TestError,
                _ => 0
            };
            worksheet.Cell(currentRow, 6).Value = double.IsNaN(col6Value) ? string.Empty : col6Value;
            if (epochResult.TrainOrTestType == EpochResult.TrainOrTest.Train || epochResult.TrainOrTestType == EpochResult.TrainOrTest.Test)
            {
                worksheet.Cell(currentRow, 7).Value = (int)(epochResult.ElapsedMilliseconds / 1000 / 60 / 60);
                worksheet.Cell(currentRow, 8).Value = (int)(epochResult.ElapsedMilliseconds / 1000 / 60 % 60);
                worksheet.Cell(currentRow, 9).Value = (int)(epochResult.ElapsedMilliseconds / 1000 % 60 % 60);
            }

            if (File.Exists(RecordFilePath))
            {
                workbook.Save();
            }
            else
            {
                workbook.SaveAs(RecordFilePath);
            }
        }

        private void WriteTemplate(int epoch, IXLWorksheet worksheet, int trainDataFileCount, int testDataFileCount)
        {
            const int headerCount = 1;
            const int trainTotalRowCount = 1;
            const int testTotalRowCount = 1;

            var firstTrainRecordRow = worksheet.Column(3).CellsUsed().Where(cell => cell.Value.ToString() == "train").FirstOrDefault()?.Address?.RowNumber ?? -1;
            var firstTestRecordRow = worksheet.Column(3).CellsUsed().Where(cell => cell.Value.ToString() == "test").FirstOrDefault()?.Address?.RowNumber ?? -1;
            var latestRecordIsCurrentEpoch = worksheet.Column(2).CellsUsed(cell => cell.Value.IsNumber).LastOrDefault()?.GetValue<int>() == epoch;
            var latestTrainRecordRow = worksheet.Column(3).CellsUsed().Where(cell => cell.Value.ToString() == "train").LastOrDefault()?.Address?.RowNumber ?? -1;
            var latestTestRecordRow = worksheet.Column(3).CellsUsed().Where(cell => cell.Value.ToString() == "test").LastOrDefault()?.Address?.RowNumber ?? -1;
            var latestTestRecordRowIsCurrentEpoch = worksheet.Column(2).CellsUsed().Any(cell => cell.Address.RowNumber == latestTestRecordRow
                                                                                             && int.TryParse(cell.Value.ToString(), out int thisEpoch));
            var currentEpochTrainRows = worksheet.Column(3).CellsUsed().Where(cell => cell.Value.ToString() == "train" && int.TryParse(cell.Worksheet.Cell(cell.Address.RowNumber, 2).Value.ToString(), out int thisEpoch) && thisEpoch == epoch);
            var currentEpochTrainTotalRows = worksheet.Column(3).CellsUsed().Where(cell => cell.Value.ToString() == "traintotal" && int.TryParse(cell.Worksheet.Cell(cell.Address.RowNumber, 2).Value.ToString(), out int thisEpoch) && thisEpoch == epoch);
            var currentEpochTestRows = worksheet.Column(3).CellsUsed().Where(cell => cell.Value.ToString() == "test" && int.TryParse(cell.Worksheet.Cell(cell.Address.RowNumber, 2).Value.ToString(), out int thisEpoch) && thisEpoch == epoch);
            var currentEpochTestTotalRows = worksheet.Column(3).CellsUsed().Where(cell => cell.Value.ToString() == "testtotal" && int.TryParse(cell.Worksheet.Cell(cell.Address.RowNumber, 2).Value.ToString(), out int thisEpoch) && thisEpoch == epoch);
            var firstRow = worksheet.FirstColumn().LastCellUsed().Address.RowNumber + (latestRecordIsCurrentEpoch ? 0 : 1);
            if (latestTrainRecordRow != -1 && latestTestRecordRow == -1)
            {
                firstRow++;
            }
            var endRow = headerCount + Epoch *  (trainDataFileCount + trainTotalRowCount + testDataFileCount + testTotalRowCount);

            //if (firstRow == endRow) return;

            var isFilledNo = worksheet.FirstColumn().LastCellUsed().Address.RowNumber >= endRow;

            if (!isFilledNo)
            {
                //No
                for (int i = firstRow; i <= endRow; i++)
                {
                    worksheet.Cell(i, 1).Value = i - 1;
                }

                //epoch
                for (int i = firstRow; i <= endRow; i++)
                {
                    worksheet.Cell(i, 2).Value = epoch;
                }

                //train or test
                for (int i = firstRow; i <= endRow; i++)
                {
                    if (latestTestRecordRow != -1 && latestTrainRecordRow != -1)
                    {

                        if (currentEpochTrainRows.Count() < trainDataFileCount)
                        {
                            if (i >= firstRow && i < firstRow + trainDataFileCount)
                            {
                                worksheet.Cell(i, 3).Value = "train";
                            }
                        }
                        else if (currentEpochTrainTotalRows.Count() < 1)
                        {
                            if (i >= firstRow + currentEpochTrainRows.Count() && i < firstRow + currentEpochTrainRows.Count() + 1)
                            {
                                worksheet.Cell(i, 3).Value = "traintotal";
                            }
                        }
                        else if (currentEpochTestRows.Count() < testDataFileCount)
                        {
                            if (i >= firstRow + currentEpochTrainRows.Count() + currentEpochTrainTotalRows.Count() && i < firstRow + currentEpochTrainRows.Count() + currentEpochTrainTotalRows.Count() + testDataFileCount)
                            {
                                worksheet.Cell(i, 3).Value = "test";
                            }
                        }
                        else if (currentEpochTestTotalRows.Count() < 1)
                        {
                            if (i >= firstRow + currentEpochTrainRows.Count() + currentEpochTrainTotalRows.Count() + currentEpochTestRows.Count() && i <= firstRow + currentEpochTrainRows.Count() + currentEpochTrainTotalRows.Count() + currentEpochTestRows.Count() + 1)
                            {
                                worksheet.Cell(i, 3).Value = "testtotal";
                            }
                        }
                    }
                    else if (latestTrainRecordRow == -1 && latestTestRecordRow == -1)
                    {
                        if (i >= firstRow && i < firstRow + trainDataFileCount)
                        {
                            worksheet.Cell(i, 3).Value = "train";
                        }
                        else if (i >= firstRow + trainDataFileCount && i < firstRow + trainDataFileCount + 1)
                        {
                            worksheet.Cell(i, 3).Value = "traintotal";
                        }
                        else if (i >= firstRow + trainDataFileCount + 1 && i < firstRow + trainDataFileCount + testDataFileCount + 1)
                        {
                            worksheet.Cell(i, 3).Value = "test";
                        }
                        else if (i >= firstRow + trainDataFileCount + testDataFileCount + 1 && i < firstRow + trainDataFileCount + testDataFileCount + 2)
                        {
                            worksheet.Cell(i, 3).Value = "testtotal";
                        }
                    }
                    else if (i >= firstTrainRecordRow && i < firstTrainRecordRow + trainDataFileCount)
                    {
                        worksheet.Cell(i, 3).Value = "train";
                    }
                    else if (i >= firstTrainRecordRow + trainDataFileCount && i < firstTrainRecordRow + trainDataFileCount + 1)
                    {
                        worksheet.Cell(i, 3).Value = "traintotal";
                        firstTestRecordRow = i + 1;
                    }
                    else if (i >= firstTestRecordRow && i < firstTestRecordRow + testDataFileCount)
                    {
                        worksheet.Cell(i, 3).Value = "test";
                    }
                    else if (i >= firstTestRecordRow + testDataFileCount && i < firstTestRecordRow + testDataFileCount + 1)
                    {
                        worksheet.Cell(i, 3).Value = "testtotal";
                    }
                }
            }

            var latestEpochCell = worksheet.Column(2).CellsUsed().LastOrDefault();
            var latestEpoch = latestEpochCell is not null ? latestEpochCell.GetValue<int>() : 0;

            var epochColumn = worksheet.Column(2);
            // 現在のエポックに対応する行番号を取得します
            var currentEpochRowNumbers = epochColumn.CellsUsed()
                                                    .Where(cell => int.TryParse(cell.GetValue<string>(), out var value) ? value == Epoch : false)
                                                    .Select(c => c.Address.RowNumber)
                                                    .OrderBy(x => x)
                                                    .ToArray();

            // "train"行をフィルタリングします
            var trainRows = currentEpochRowNumbers.Take(trainDataFileCount).ToArray();
            // "test"行をフィルタリングします
            var testRows = currentEpochRowNumbers.Skip(trainDataFileCount + 1).Take(testDataFileCount).ToArray();

            foreach (var cell in trainRows.Select(row => worksheet.Cell(row, 4)))
            {
                cell.Value = (TrainSet as MovieFileDataset).MovieFilePaths.ElementAt((TrainLoader as MovieFileDataLoader).MovieIndex[cell.Address.RowNumber - trainRows.FirstOrDefault()].GetData<int>());
            }

            foreach (var cell in testRows.Select(row => worksheet.Cell(row, 4)))
            {
                cell.Value = (TestSet as MovieFileDataset).MovieFilePaths.ElementAt((TestLoader as MovieFileDataLoader).MovieIndex[cell.Address.RowNumber - testRows.FirstOrDefault()].GetData<int>());
            }
        }

        private static int GetNextNo(IXLWorksheet worksheet)
        {
            var firstColumn_lastCell = worksheet.Column(5).LastCellUsed();
            if (firstColumn_lastCell is not null)
            {
                return firstColumn_lastCell.Address.RowNumber;
            }
            throw new InvalidOperationException();
        }

        private void WriteHorizontalResult(EpochResult epochResult)
        {
            using var workbook = File.Exists(RecordFilePath) ? new XLWorkbook(RecordFilePath) : new XLWorkbook();
            var worksheet = workbook.Worksheets.SingleOrDefault(s => s.Name == "data") ?? workbook.AddWorksheet("data");
            worksheet.Cell(1, 1).Value = "epoch";
            switch (epochResult.ModelType)
            {
                case ModelType.Regression:
                    worksheet.Cell(2, 1).Value = "train_loss";
                    worksheet.Cell(3, 1).Value = "train_error";
                    worksheet.Cell(4, 1).Value = "test_loss";
                    worksheet.Cell(5, 1).Value = "test_error";
                    break;
                case ModelType.Classification:
                    worksheet.Cell(2, 1).Value = "train_loss";
                    worksheet.Cell(3, 1).Value = "train_accuracy";
                    worksheet.Cell(4, 1).Value = "test_loss";
                    worksheet.Cell(5, 1).Value = "test_accuracy";
                    break;
            }
            worksheet.Cell(6, 1).Value = "h";
            worksheet.Cell(7, 1).Value = "m";
            worksheet.Cell(8, 1).Value = "s";

            worksheet.Cell(1, epochResult.Epoch + 1).Value = epochResult.Epoch;
            switch (epochResult.ModelType)
            {
                case ModelType.Regression:
                    worksheet.Cell(2, epochResult.Epoch + 1).Value = epochResult.TrainLoss;
                    worksheet.Cell(3, epochResult.Epoch + 1).Value = epochResult.TrainError;
                    worksheet.Cell(4, epochResult.Epoch + 1).Value = epochResult.TestLoss;
                    worksheet.Cell(5, epochResult.Epoch + 1).Value = epochResult.TestError;
                    break;
                case ModelType.Classification:
                    worksheet.Cell(2, epochResult.Epoch + 1).Value = epochResult.TrainLoss;
                    worksheet.Cell(3, epochResult.Epoch + 1).Value = epochResult.TrainAccuracy;
                    worksheet.Cell(4, epochResult.Epoch + 1).Value = epochResult.TestLoss;
                    worksheet.Cell(5, epochResult.Epoch + 1).Value = epochResult.TestAccuracy;
                    break;
            }
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
        }

        /// <summary>
        /// CalcLossメソッドにて計算された損失に対して追加の損失を計算します.
        /// </summary>
        /// <param name="loss">CalcLossメソッドにて計算された損失</param>
        /// <returns>追加の損失を含めた損失</returns>
        public virtual Variable CalcAdditionalLoss(Variable loss)
        {
            return loss + L2Regularization.Invoke(Model.Params(), new NDarray(0.01).ToVariable())[0];
        }

        /// <summary>
        /// 損失を計算します.
        /// </summary>
        /// <remarks>
        /// <para>損失関数の種類：</para>
        /// <para> 1. 回帰モデル:</para>
        /// <para>  - 平均二乗誤差 (MSE: Mean Squared Error) 予測値と正解値の二乗誤差の平均を計算します.最もシンプルな損失関数の一つであり,計算コストも比較的安価です.ただし,外れ値の影響を受けやすいという欠点があります.</para>
        /// <para>  - 平均絶対誤差 (MAE: Mean Absolute Error) 予測値と正解値の絶対誤差の平均を計算します.MSEと異なり,外れ値の影響を受けにくいという利点があります.</para>
        /// <para>  - ハーバー損失 (Huber Loss) MSEとMAEを組み合わせたような損失関数です.小さい誤差に対してはMSE,大きい誤差に対してはMAEのように振る舞います.外れ値の影響を受けにくく,かつロバストなモデル学習に適しています.</para>
        /// <para>  </para>
        /// <para> 2. 分類モデル:</para>
        /// <para>  - 交差エントロピー損失 (Cross-Entropy Loss) 予測確率と正解ラベルの情報エントロピーを計算します.確率分布における誤差を表現するのに適しており,多クラス分類によく用いられます.</para>
        /// <para>  - ヒンジ損失 (Hinge Loss) 予測値と正解ラベルの誤差の大きさを計算します.誤分類ペナルティを導入することで,サポートベクターマシンのような最大マージン分類器の学習に適しています.</para>
        /// <para>  - 対数損失 (Log Loss) ロジスティック回帰のような二値分類において用いられる損失関数です.交差エントロピー損失と類似していますが,計算コストが若干安価です.</para>
        /// <para>  </para>
        /// <para> 3. 生成モデル:</para>
        /// <para>  - 最尤推定 (Maximum Likelihood Estimation) 生成モデルのパラメータを決定するために,データの生成確率を最大化するようにパラメータを更新します.シンプルな方法ですが,計算コストが高くなる場合があるという欠点があります.</para>
        /// <para>  - 変分下限 (ELBO: Evidence Lower BOund) 最尤推定の欠点を補うために用いられる方法です.積分計算を用いて近似的に最尤推定を行うことで,計算コストを削減することができます.</para>
        /// <para>  - 敵対的損失 (Adversarial Loss) 生成モデルと識別モデルを用いて互いに競い合わせることで,生成モデルを学習させる方法です.近年,画像生成や自然言語処理などの分野で注目を集めています.</para>
        /// </remarks>
        /// <param name="y">モデルからの出力（予測値）</param>
        /// <param name="t">グラウンドトゥルース（Ground Truth）</param>
        /// <returns></returns>
        public virtual Variable CalcLoss(Variable y, NDarray t)
        {
            return SoftmaxCrossEntropy.Invoke(y, t.ToVariable())[0];
        }

        /// <summary>
        /// 精度を計算します
        /// </summary>
        /// <param name="y">モデルからの出力（予測値）</param>
        /// <param name="t">グラウンドトゥルース（Ground Truth）</param>
        /// <returns></returns>
        public virtual Variable CalcAccuracy(Variable y, NDarray t)
        {
            var accuracy = new Accuracy();
            return accuracy.Call(Params.New.SetKeywordArg(y, t))[0];
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
