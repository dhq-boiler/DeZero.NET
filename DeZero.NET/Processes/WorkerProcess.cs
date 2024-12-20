﻿using ClosedXML.Excel;
using DeZero.NET.Core;
using DeZero.NET.Datasets;
using DeZero.NET.Extensions;
using DeZero.NET.Functions;
using DeZero.NET.LearningRateSchedulers;
using DeZero.NET.Log;
using DeZero.NET.Models;
using DeZero.NET.Optimizers;
using DeZero.NET.Recorder;
using Python.Runtime;
using System.Diagnostics;
using System.Text;
using DeZero.NET.Monitors;

namespace DeZero.NET.Processes
{
    /// <summary>
    /// Abstract class for worker processes.
    /// </summary>
    public abstract class WorkerProcess
    {
        private static ILogger _logger { get; set; }

        public static LogLevel MinimumLogLevel { get; set; } = LogLevel.Info;

        public bool IsVerbose { get; set; } = false;

        /// <summary>
        /// Gets the arguments for the worker process.
        /// </summary>
        public object[] Args { get; }

        /// <summary>
        /// Gets or sets the number of epochs for training.
        /// </summary>
        public int Epoch { get; set; }

        /// <summary>
        /// Gets or sets the batch size for training.
        /// </summary>
        public int BatchSize { get; set; }

        /// <summary>
        /// Gets or sets the size of the hidden layers.
        /// </summary>
        public int HiddenSize { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether GPU is enabled.
        /// </summary>
        public bool EnableGpu { get; set; }

        /// <summary>
        /// Gets or sets the file path for recording results.
        /// </summary>
        public string RecordFilePath { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether to dispose all inputs.
        /// </summary>
        public bool DisposeAllInputs { get; set; } = false;

        /// <summary>
        /// Gets the training dataset.
        /// </summary>
        public DeZero.NET.Datasets.Dataset TrainSet { get; private set; }

        /// <summary>
        /// Gets the test dataset.
        /// </summary>
        public DeZero.NET.Datasets.Dataset TestSet { get; private set; }

        /// <summary>
        /// Gets the data provider for the training dataset.
        /// </summary>
        public DeZero.NET.Datasets.IDataProvider TrainLoader { get; private set; }

        /// <summary>
        /// Gets the data provider for the test dataset.
        /// </summary>
        public DeZero.NET.Datasets.IDataProvider TestLoader { get; private set; }

        /// <summary>
        /// Gets the model used for training and testing.
        /// </summary>
        public Models.Model Model { get; private set; }

        /// <summary>
        /// Gets the optimizer used for training the model.
        /// </summary>
        public Optimizer Optimizer { get; private set; }

        /// <summary>
        /// Gets the learning rate manager.
        /// </summary>
        public LearningRateManager LearningRateManager { get; private set; }

        /// <summary>
        /// Gets the loss plotter.
        /// </summary>
        public RealtimeLossPlotter LossPlotter { get; private set; }

        /// <summary>
        /// Gets the type of the model.
        /// </summary>
        public abstract ModelType ModelType { get; }

        /// <summary>
        /// python311.dll のパスを取得します.
        /// </summary>
        /// <remarks>サブクラスで実装します.</remarks>
        public abstract string PythonDLLPath { get; }

        public WorkerProcess(LogLevel minimumLogLevel = LogLevel.Info)
        { 
            AppDomain.CurrentDomain.ProcessExit += (sender, e) =>
            {
                Environment.Exit(-1);
            };

            AppDomain.CurrentDomain.UnhandledException += (sender, eventArgs) =>
            {
                var ex = (Exception)eventArgs.ExceptionObject;
                File.WriteAllText("worker-crash.log", $"""
        Time: {DateTime.Now}
        Exception: {ex.GetType().Name}:{ex.Message}
        InnerException: {ex.InnerException?.GetType().Name}:{ex.InnerException?.Message}
        StackTrace: {ex.StackTrace}
        Source: {ex.Source}
        TargetSite: {ex.TargetSite}
        """);
                //シャットダウンを申請する
                Console.WriteLine("__SHUTDOWN__");
            };

            MinimumLogLevel = minimumLogLevel;
            _logger = new ConsoleLogger(minimumLogLevel, isVerbose: false);

            Console.OutputEncoding = Encoding.UTF8;
            Args = Environment.GetCommandLineArgs().Skip(1).ToArray();
            InitializeArguments(Args);
            InitializePython();
            InitializeXp();
            SetGpuUse();
        }

        /// <summary>
        /// Initializes the xp class.
        /// </summary>
        private void InitializeXp()
        {
            using var progress = _logger.BeginProgress("xp.Initialize...");
            try
            {
                xp.Initialize();
                progress.Complete();
            }
            catch (Exception ex)
            {
                progress.Failed(ex.Message);
                Environment.Exit(-1);
            }
        }

        /// <summary>
        /// Sets the training dataset.
        /// </summary>
        /// <param name="trainSet">A function that returns the training dataset.</param>
        public void SetTrainSet(Func<DeZero.NET.Datasets.Dataset> trainSet)
        {
            using var progress = _logger.BeginProgress("Start preparing train_set...");
            try
            {
                TrainSet = trainSet();
                progress.Complete();
            }
            catch (Exception ex)
            {
                progress.Failed(ex.Message);
                Environment.Exit(-1);
            }
        }

        /// <summary>
        /// Sets the test dataset.
        /// </summary>
        /// <param name="testSet">A function that returns the test dataset.</param>
        public void SetTestSet(Func<DeZero.NET.Datasets.Dataset> testSet)
        {
            using var progress = _logger.BeginProgress("Start preparing test_set...");
            try
            {
                TestSet = testSet();
                progress.Complete();
            }
            catch (Exception ex)
            {
                progress.Failed(ex.Message);
                Environment.Exit(-1);
            }
        }

        /// <summary>
        /// Sets the data provider for the training dataset.
        /// </summary>
        /// <param name="trainLoader">A function that returns the data provider for the training dataset.</param>
        public void SetTrainLoader(Func<DeZero.NET.Datasets.Dataset, int, DeZero.NET.Datasets.IDataProvider> trainLoader)
        {
            using var progress = _logger.BeginProgress("Start preparing train_loader...");
            try
            {
                TrainLoader = trainLoader(this.TrainSet, BatchSize);
                TrainLoader.WorkerProcess = this;
                TrainLoader.OnSwitchDataFile = (resultMetrics, movie_file_path, sw) =>
                {
                    if (double.IsNaN(resultMetrics.SumLoss / TrainLoader.Length) || double.IsNaN(resultMetrics.SumAccuracy / TrainLoader.Length) 
                                                                                 || double.IsNaN(resultMetrics.SumError/ TrainLoader.Length))
                    {
                        _logger.LogError("NaN detected in metrics.");
                        Console.WriteLine("__SHUTDOWN__");
                        Environment.Exit(-1);
                    }

                    ConsoleOutWriteLinePastProcess(TrainOrTest.Train, resultMetrics.SumLoss / TrainLoader.Length, resultMetrics.SumError / TrainLoader.Length, resultMetrics.SumAccuracy / TrainLoader.Length);

                    EpochResult epochResult = new EpochResult
                    {
                        ModelType = ModelType,
                        Epoch = Epoch,
                        TargetDataFile = movie_file_path,
                        TrainOrTestType = EpochResult.TrainOrTest.Train,
                        TrainLoss = resultMetrics.SumLoss / TrainLoader.Length,
                        TrainError = resultMetrics.SumError / TrainLoader.Length,
                        TrainAccuracy = resultMetrics.SumAccuracy / TrainLoader.Length,
                        ElapsedMilliseconds = sw.ElapsedMilliseconds
                    };
                    WriteResultToRecordFile(epochResult);
                    sw.Stop();
                    sw.Reset();

                    _weightsAreDirty = true;
                    resultMetrics.Initialize();

                    sw.Start();
                };
                progress.Complete();
            }
            catch (Exception ex)
            {
                progress.Failed(ex.Message);
                Environment.Exit(-1);
            }
        }

        /// <summary>
        /// Sets the data provider for the test dataset.
        /// </summary>
        /// <param name="testLoader">A function that returns the data provider for the test dataset.</param>
        public void SetTestLoader(Func<DeZero.NET.Datasets.Dataset, int, DeZero.NET.Datasets.IDataProvider> testLoader)
        {
            using var progress = _logger.BeginProgress("Start preparing test_loader...");
            try
            {
                TestLoader = testLoader(this.TestSet, BatchSize);
                TestLoader.OnSwitchDataFile = (resultMetrics, movie_file_path, sw) =>
                {
                    if (double.IsNaN(resultMetrics.SumLoss / TestLoader.Length) || double.IsNaN(resultMetrics.SumAccuracy / TestLoader.Length)
                                                                                 || double.IsNaN(resultMetrics.SumError / TestLoader.Length))
                    {
                        _logger.LogError("NaN detected in metrics.");
                        Console.WriteLine("__SHUTDOWN__");
                        Environment.Exit(-1);
                    }

                    ConsoleOutWriteLinePastProcess(TrainOrTest.Test, resultMetrics.SumLoss / TestLoader.Length, resultMetrics.SumError / TestLoader.Length, resultMetrics.SumAccuracy / TestLoader.Length);

                    var epochResult = new EpochResult
                    {
                        ModelType = ModelType,
                        Epoch = Epoch,
                        TargetDataFile = movie_file_path,
                        TrainOrTestType = EpochResult.TrainOrTest.Test,
                        TestLoss = resultMetrics.SumLoss / TestLoader.Length,
                        TestError = resultMetrics.SumError / TestLoader.Length,
                        TestAccuracy = resultMetrics.SumAccuracy / TestLoader.Length,
                        ElapsedMilliseconds = sw.ElapsedMilliseconds
                    };
                    WriteResultToRecordFile(epochResult);
                    sw.Stop();
                    sw.Reset();

                    resultMetrics.Initialize();

                    sw.Start();
                };
                progress.Complete();
            }
            catch (Exception ex)
            {
                _logger.LogError($"Failed to prepare test_loader: {ex.Message}");
                Environment.Exit(-1);
            }
        }

        /// <summary>
        /// Sets the model used for training and testing.
        /// </summary>
        /// <param name="model">A function that returns the model.</param>
        public void SetModel(Func<Models.Model> model)
        {
            using var progress = _logger.BeginProgress("Start preparing model...");
            try
            {
                Model = model();
                progress.Complete();
            }
            catch (Exception ex)
            {
                progress.Failed($"Failed to prepare model: {ex.Message}");
                Environment.Exit(-1);
            }
        }

        /// <summary>
        /// Loads existing weights into the model.
        /// </summary>
        public virtual void LoadExistedWeights()
        {
            Directory.CreateDirectory("weights");
            if (Directory.EnumerateFiles("weights").Any())
            {
                using var progress = _logger.BeginProgress("Start loading weights...");
                try
                {
                    using (new Gpu.TemporaryDisable())
                    {
                        Model.LoadWeights();
                    }
                    progress.Complete();
                }
                catch (Exception ex)
                {
                    progress.Failed($"Failed to load weights: {ex.Message}");
                    Environment.Exit(-1);
                }
            }
        }

        /// <summary>
        /// Saves the current weights of the model.
        /// </summary>
        public void SaveWeights()
        {

            using var progress = _logger.BeginProgress("Save weights...");
            try
            {
                Model.SaveWeights();
                progress.Complete();
            }
            catch (Exception ex)
            {
                progress.Failed($"Failed to save weights: {ex.Message}");
                Environment.Exit(-1);
            }
        }

        /// <summary>
        /// Sets the optimizer used for training the model.
        /// </summary>
        /// <param name="optimizer">A function that returns the optimizer.</param>
        public void SetOptimizer(Func<Models.Model, Optimizer> optimizer)
        {
            using var progress = _logger.BeginProgress("Start preparing optimizer...");
            try
            {
                Optimizer = optimizer(Model);
                if (LearningRateManager is not null)
                {
                    Optimizer.SetNewLr(LearningRateManager.GetInitialLearningRate(Epoch));
                }
                progress.Complete();
            }
            catch (Exception ex)
            {
                progress.Failed($"Failed to prepare optimizer: {ex.Message}");
                Environment.Exit(-1);
            }
        }

        /// <summary>
        /// Loads the optimizer state.
        /// </summary>
        public void LoadOptimizer()
        {
            Directory.CreateDirectory("optimizer");
            if (Directory.EnumerateFiles("optimizer").Any())
            {
                using var progress = _logger.BeginProgress("Start optimizer states...");
                try
                {
                    using (new Gpu.TemporaryDisable())
                    {
                        Optimizer.LoadParameters();
                    }
                    progress.Complete();
                }
                catch (Exception ex)
                {
                    progress.Failed($"Failed to load optimizer states: {ex.Message}");
                    Environment.Exit(-1);
                }
            }
        }

        /// <summary>
        /// Saves the optimizer state.
        /// </summary>
        public void SaveOptimizer()
        {
            using var progress = _logger.BeginProgress("Save optimizer states...");
            try
            {
                Optimizer.SaveParameters();
                progress.Complete();
            }
            catch (Exception ex)
            {
                progress.Failed($"Failed to save optimizer states: {ex.Message}");
                Environment.Exit(-1);
            }
        }

        public void SetLearningRateScheduler(Func<ILearningRateScheduler> value, float initialLr)
        {
            using var progress = _logger.BeginProgress("Start preparing LearningRateScheduler...");
            try
            {
                LearningRateManager = new LearningRateManager(value(), initialLr, _logger);
                if (Optimizer is not null)
                {
                    Optimizer.SetNewLr(LearningRateManager.GetInitialLearningRate(Epoch));
                }
                progress.Complete();
            }
            catch (Exception ex)
            {
                progress.Failed($"Failed to prepare LearningRateScheduler: {ex.Message}");
                Environment.Exit(-1);
            }
        }

        public void InitializeLossPlotter()
        {
            using var progress = _logger.BeginProgress("Start initializing LossPlotter...");
            try
            {
                LossPlotter = new RealtimeLossPlotter(MaxEpoch, BatchSize);
                LossPlotter.LoadColorPalette("colors.txt");

                var currentFile =
                    (this.TrainSet as MovieFileDataset).MovieFilePaths[this.TrainLoader.CurrentMovieIndex];

                LossPlotter.LoadLoss($"{Path.GetFileNameWithoutExtension(currentFile)}", Epoch);
                progress.Complete();
            }
            catch (Exception ex)
            {
                progress.Failed($"Failed to initializing LossPlotter: {ex.Message}");
                Environment.Exit(-1);
            }
        }

        /// <summary>
        /// Resumes the state of the training process.
        /// </summary>
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
            using var progress = _logger.BeginProgress("Resume state...");

            try
            {
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

                progress.Complete();
            }
            catch (Exception ex)
            {
                progress.Failed($"Failed to resume state: {ex.Message}");
                Environment.Exit(-1);
            }
        }

        /// <summary>
        /// Initializes the arguments for the worker process.
        /// </summary>
        /// <param name="args">The arguments to initialize.</param>
        protected abstract void InitializeArguments(object[] args);

        protected virtual Func<NDarray, long> UnitLength => (t) => t.len;

        public float CurrentLoss { get; private set; }

        public float CurrentError { get; private set; }

        public int MaxEpoch { get; protected set; }

        private bool _weightsAreDirty = false;

        /// <summary>
        /// Runs the worker process.
        /// </summary>
        public void Run()
        {
            using (this.TrackMemory("Training Epoch", verbose: true))
            {
                var resultMetrics = new ResultMetrics();
                var count = 0;

                Stopwatch sw = new Stopwatch();

                _logger.LogInfo($"epoch : {Epoch}");
                _logger.LogInfo($"training...");

                sw.Start();

                if (VRAMLeakDetector.IsEnabled)
                {
                    VRAMLeakDetector.StartMemoryMonitoring();
                }

                TrainLoader.SetResultMetricsAndStopwatch(resultMetrics, sw);

                foreach (var (x, t) in TrainLoader)
                {
                    if (VRAMLeakDetector.IsEnabled)
                    {
                        Console.WriteLine(VRAMLeakDetector.GetAllocationReport());
                    }

                    if (PythonObjectTracker.IsEnabled)
                    {
                        Console.WriteLine(PythonObjectTracker.GenerateReport());
                    }

                    if (VRAMLeakDetector.IsEnabled)
                    {
                        VRAMLeakDetector.DetectPotentialLeaks();
                        VRAMLeakDetector.Iteration++;
                    }

                    using var forwardScope = new BatchScope();
                    using (this.TrackMemory($"Batch {count}"))
                    {
                        var localSw = new Stopwatch();
                        localSw.Start();
                        try
                        {
                            using var y = Model.Call(x.ToVariable())[0];
                            GpuMemoryMonitor.Instance.LogMemoryUsage("After Model.Call");

                            forwardScope.TrackTemporary(y);
                            using var loss = CalcLoss(y, t);
                            using var evalValue = CalcEvaluationMetric(y, t);
                            using var total_loss = CalcAdditionalLoss(loss);

                            Model.ClearGrads();
                            Optimizer.ClearGrads();

                            GpuMemoryMonitor.Instance.LogMemoryUsage("Before Backward");

                            total_loss.Backward(retain_grad: true, initializeGrad: true);

                            GpuMemoryMonitor.Instance.LogMemoryUsage("After Backward");

                            CurrentLoss = total_loss.Data.Value.asscalar<float>();
                            CurrentError = evalValue.Data.Value.asscalar<float>();

                            if (LossPlotter is not null)
                            {
                                LossPlotter.Update((int)TrainLoader.CurrentFrameIndex, CurrentLoss, CurrentError, Optimizer.Lr, Epoch);
                            }

                            resultMetrics.SumLoss += total_loss.Data.Value.asscalar<float>() * UnitLength(t);

                            switch (ModelType)
                            {
                                case ModelType.Regression:
                                    resultMetrics.SumError += evalValue.Data.Value.asscalar<float>() * UnitLength(t);
                                    break;
                                case ModelType.Classification:
                                    resultMetrics.SumAccuracy += evalValue.Data.Value.asscalar<float>() * UnitLength(t);
                                    break;
                            }

                            Optimizer.Update(null);

                            //loss.CleanupComputationalGraph(); // 計算グラフのクリーンアップ
                            evalValue.CleanupComputationalGraph(); // 計算グラフのクリーンアップ
                            total_loss.CleanupComputationalGraph(); // 計算グラフのクリーンアップ

                            if (DisposeAllInputs)
                            {
                                Model.DisposeAllInputs();
                            }
                            count++;

                            //if (count % 10 == 0)  // 10バッチごとにメモリプールをクリア
                            //{
                            //    GpuMemoryMonitor.ForceMemoryPool();
                            //}

                            x.Dispose();
                            t.Dispose();
                            GC.Collect();
                            Finalizer.Instance.Collect();
                        }
                        catch (OutOfMemoryException)
                        {
                            _logger.LogError("Out of memory detected. Training will exit.");
                            throw;
                        }
                        finally
                        {
                            GpuMemoryMonitor.Instance.CleanupMemory();
                            localSw.Stop();
                            TrainLoader.SetLocalStopwatch(localSw);
                        }
                    }
                }


                EpochResult epochResult = new EpochResult
                {
                    ModelType = ModelType,
                    Epoch = Epoch,
                    TrainOrTestType = EpochResult.TrainOrTest.TrainTotal,
                    TrainLoss = resultMetrics.SumLoss / TrainLoader.Length,
                    TrainError = resultMetrics.SumError / TrainLoader.Length,
                    TrainAccuracy = resultMetrics.SumAccuracy / TrainLoader.Length,
                    ElapsedMilliseconds = sw.ElapsedMilliseconds
                };

                if (double.IsNaN(epochResult.TrainLoss))
                {
                    _logger.LogInfo("skip.");
                }
                else
                {
                    _weightsAreDirty = true;
                    ConsoleOutWriteLinePastProcess(TrainOrTest.Train, resultMetrics.SumLoss / TrainLoader.Length, resultMetrics.SumError / TrainLoader.Length, resultMetrics.SumAccuracy / TrainLoader.Length);
                    WriteResultToRecordFile(epochResult);
                }
            }

            Console.WriteLine();
            _logger.LogInfo($"testing...");

            using (this.TrackMemory("Testing Epoch", verbose: true))
            {
                var count = 0;
                var test_resultMetrics = new ResultMetrics();

                Stopwatch sw = new Stopwatch();

                TestLoader.SetResultMetricsAndStopwatch(test_resultMetrics, sw);

                using (var config = ConfigExtensions.NoGrad())
                {
                    foreach (var (x, t) in TestLoader)
                    {
                        if (VRAMLeakDetector.IsEnabled)
                        {
                            Console.WriteLine(VRAMLeakDetector.GetAllocationReport());
                        }

                        if (PythonObjectTracker.IsEnabled)
                        {
                            Console.WriteLine(PythonObjectTracker.GenerateReport());
                        }

                        if (VRAMLeakDetector.IsEnabled)
                        {
                            VRAMLeakDetector.DetectPotentialLeaks();
                            VRAMLeakDetector.Iteration++;
                        }

                        using var forwardScope = new BatchScope();
                        var localSw = new Stopwatch();
                        localSw.Start();
                        try
                        {
                            using var y = Model.Call(x.ToVariable())[0];
                            forwardScope.TrackTemporary(y);

                            // モデル出力のNaNチェック
                            float[] yData = y.Data.Value.flatten().GetData<float[]>();
                            if (yData.Any(float.IsNaN))
                            {
                                _logger.LogWarning("Warning: NaN values detected in model output");
                                continue;
                            }

                            using var loss = CalcLoss(y, t);
                            float lossValue = loss.Data.Value.asscalar<float>();
                            if (float.IsNaN(lossValue))
                            {
                                _logger.LogWarning($"Warning: NaN values detected in loss calculation. Model output range: {yData.Min()} to {yData.Max()}");
                                continue;
                            }
                            loss.CleanupComputationalGraph(); // 計算グラフのクリーンアップ

                            using var evalValue = CalcEvaluationMetric(y, t);
                            float evalValue_float = evalValue.Data.Value.asscalar<float>();
                            if (float.IsNaN(evalValue_float))
                            {
                                _logger.LogWarning("Warning: NaN values detected in evaluation metric");
                                continue;
                            }
                            evalValue.CleanupComputationalGraph(); // 計算グラフのクリーンアップ

                            // すべての値が正常な場合のみメトリクスを更新
                            test_resultMetrics.SumLoss += lossValue * UnitLength(t);
                            switch (ModelType)
                            {
                                case ModelType.Regression:
                                    test_resultMetrics.SumError += evalValue_float * UnitLength(t);
                                    break;
                                case ModelType.Classification:
                                    test_resultMetrics.SumAccuracy += evalValue_float * UnitLength(t);
                                    break;
                            }

                            if (DisposeAllInputs)
                            {
                                Model.DisposeAllInputs();
                            }

                            count++;

                            if (count % 10 == 0)  // 10バッチごとにメモリプールをクリア
                            {
                                GpuMemoryMonitor.ForceMemoryPool();
                            }

                            x.Dispose();
                            t.Dispose();
                            GC.Collect();
                            Finalizer.Instance.Collect();
                        }
                        catch (OutOfMemoryException)
                        {
                            _logger.LogError("Out of memory detected. Testing will exit.");
                            throw;
                        }
                        finally
                        {
                            localSw.Stop();
                            TestLoader.SetLocalStopwatch(localSw);
                        }
                    }
                }

                sw.Stop();

                // メトリクスの集計時にNaN値の影響を考慮
                var testLength = TestLoader.Length;
                if (testLength > 0 && test_resultMetrics.SumLoss > 0) // メトリクスが正常に集計されている場合のみ
                {
                    test_resultMetrics.SumLoss /= testLength;
                    test_resultMetrics.SumError /= testLength;
                    test_resultMetrics.SumAccuracy /= testLength;

                    ConsoleOutWriteLinePastProcess(
                        TrainOrTest.Test,
                        test_resultMetrics.SumLoss,
                        test_resultMetrics.SumError,
                        test_resultMetrics.SumAccuracy
                    );
                }
                else
                {
                    _logger.LogWarning("Warning: Unable to calculate test metrics due to invalid values");
                }

                _logger.LogInfo($"time : {(int)(sw.ElapsedMilliseconds / 1000 / 60)}m{(sw.ElapsedMilliseconds / 1000 % 60)}s");
                _logger.LogInfo("==================================================================================");

                var epochResult = new EpochResult
                {
                    ModelType = ModelType,
                    Epoch = Epoch,
                    TrainOrTestType = EpochResult.TrainOrTest.TestTotal,
                    TestLoss = test_resultMetrics.SumLoss / TestLoader.Length,
                    TestError = test_resultMetrics.SumError / TestLoader.Length,
                    TestAccuracy = test_resultMetrics.SumAccuracy / TestLoader.Length,
                    ElapsedMilliseconds = sw.ElapsedMilliseconds
                };
                WriteResultToRecordFile(epochResult);
            }
            SaveWeights();
            SaveOptimizer();
            ExitSequence();
        }

        /// <summary>
        /// Writes the process information to the console.
        /// </summary>
        /// <param name="trainOrTest">Indicates whether it is training or testing.</param>
        /// <param name="loss">The loss value.</param>
        /// <param name="err">The error value.</param>
        /// <param name="acc">The accuracy value.</param>
        private void ConsoleOutWriteLinePastProcess(TrainOrTest trainOrTest, double loss, double err, double acc)
        {
            var title = trainOrTest switch
            {
                TrainOrTest.Train => "train",
                TrainOrTest.Test => "test",
                _ => string.Empty
            };

            switch (ModelType)
            {
                case ModelType.Regression:
                    Console.WriteLine($"{title} loss: {loss}, error: {err}");
                    break;
                case ModelType.Classification:
                    Console.WriteLine($"{title} loss: {loss}, accuracy: {acc}");
                    break;
            }
        }

        private enum TrainOrTest
        {
            Train,
            Test
        }

        /// <summary>
        /// Executes the exit sequence for the worker process.
        /// </summary>
        private void ExitSequence()
        {
            //親プロセスに終了を通知
            File.Create("signal").Close();

            //親プロセスからKillされるか,自害するか
            Environment.Exit(0);
        }

        /// <summary>
        /// Writes the result to the record file.
        /// </summary>
        /// <param name="epochResult">The result of the epoch.</param>
        private void WriteResultToRecordFile(EpochResult epochResult)
        {
            using var progress = _logger.BeginProgress($"Save XLSX:{RecordFilePath} ...");
            try
            {
                if (this.TrainLoader is MovieFileDataLoader && TestLoader is MovieFileDataLoader)
                {
                    WriteVerticalResult(epochResult);
                }
                else
                {
                    WriteHorizontalResult(epochResult);
                }
                progress.Complete();
            }
            catch (Exception ex)
            {
                progress.Failed($"Failed to save XLSX: {ex.Message}");
                Environment.Exit(-1);
            }
        }

        /// <summary>
        /// Writes the vertical result to the record file.
        /// </summary>
        /// <param name="epochResult">The result of the epoch.</param>
        private void WriteVerticalResult(EpochResult epochResult)
        {
            using var workbook = File.Exists(RecordFilePath) ? new XLWorkbook(RecordFilePath) : new XLWorkbook();
            var worksheet = workbook.Worksheets.SingleOrDefault(s => s.Name == "data") ?? workbook.AddWorksheet("data");

            //ヘッダー行を書き込みます
            worksheet.Cell(1, 1).Value = "No";
            worksheet.Cell(1, 2).Value = "epoch";
            worksheet.Cell(1, 3).Value = "train or test";
            worksheet.Cell(1, 4).Value = "movie file";
            worksheet.Cell(1, 5).Value = "loss";
            worksheet.Cell(1, 6).Value = "error";
            worksheet.Cell(1, 7).Value = "h";
            worksheet.Cell(1, 8).Value = "m";
            worksheet.Cell(1, 9).Value = "s";

            // 現在のエポックの範囲の No, epoch, train or test, movie fileカラムを書き込みます
            WriteTemplate(epochResult.Epoch, worksheet, (TrainSet as MovieFileDataset).MovieFilePaths.Count(), (TestSet as MovieFileDataset).MovieFilePaths.Count());

            //カラム５の最初に空白があるセルの行を取得します
            
            var firstEmptyCell = FindFirstEmptyCell(worksheet, 5);
            var currentRow = firstEmptyCell?.Row ?? 2;

            // loss カラムに値を書き込みます
            var col5Value = epochResult.TrainOrTestType switch
            {
                EpochResult.TrainOrTest.Train => epochResult.TrainLoss,
                EpochResult.TrainOrTest.TrainTotal => epochResult.TrainLoss,
                EpochResult.TrainOrTest.Test => epochResult.TestLoss,
                EpochResult.TrainOrTest.TestTotal => epochResult.TestLoss,
                _ => 0
            };
            worksheet.Cell(currentRow, 5).Value = double.IsNaN(col5Value) ? string.Empty : col5Value;

            // error カラムに値を書き込みます
            var col6Value = epochResult.TrainOrTestType switch
            {
                EpochResult.TrainOrTest.Train => epochResult.TrainError,
                EpochResult.TrainOrTest.TrainTotal => epochResult.TrainError,
                EpochResult.TrainOrTest.Test => epochResult.TestError,
                EpochResult.TrainOrTest.TestTotal => epochResult.TestError,
                _ => 0
            };
            worksheet.Cell(currentRow, 6).Value = double.IsNaN(col6Value) ? string.Empty : col6Value;

            // h, m, s カラムに値を書き込みます
            if (epochResult.TrainOrTestType == EpochResult.TrainOrTest.Train || epochResult.TrainOrTestType == EpochResult.TrainOrTest.Test)
            {
                worksheet.Cell(currentRow, 7).Value = (int)(epochResult.ElapsedMilliseconds / 1000 / 60 / 60);
                worksheet.Cell(currentRow, 8).Value = (int)(epochResult.ElapsedMilliseconds / 1000 / 60 % 60);
                worksheet.Cell(currentRow, 9).Value = (int)(epochResult.ElapsedMilliseconds / 1000 % 60 % 60);
            }

            // ワークブックを保存します
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
        /// Finds the first empty cell in the specified column of the worksheet.
        /// </summary>
        /// <param name="worksheet">The worksheet to search.</param>
        /// <param name="targetColumn">The target column to search.</param>
        /// <returns>The row and column of the first empty cell, or null if none found.</returns>
        private static (int Row, int Column)? FindFirstEmptyCell(IXLWorksheet worksheet, int targetColumn)
        {
            var column = worksheet.Column(targetColumn);

            // 使用されている範囲の最後の行を取得
            var lastRow = worksheet.LastRowUsed().RowNumber();

            // 1行目から最後の行まで検索
            for (int row = 1; row <= lastRow; row++)
            {
                var cell = worksheet.Cell(row, targetColumn);

                // セルが空白かどうかをチェック
                if (string.IsNullOrWhiteSpace(cell.GetString()))
                {
                    return (row, targetColumn);
                }
            }

            // 空白セルが見つからない場合
            return null;
        }

        /// <summary>
        /// Writes the template to the worksheet.
        /// </summary>
        /// <param name="epoch">The current epoch.</param>
        /// <param name="worksheet">The worksheet to write to.</param>
        /// <param name="trainDataFileCount">The number of training data files.</param>
        /// <param name="testDataFileCount">The number of test data files.</param>
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
                cell.Value = (TrainSet as MovieFileDataset).MovieFilePaths.ElementAt((TrainLoader as MovieFileDataLoader).MovieIndex[cell.Address.RowNumber - trainRows.FirstOrDefault()].asscalar<int>());
            }

            foreach (var cell in testRows.Select(row => worksheet.Cell(row, 4)))
            {
                cell.Value = (TestSet as MovieFileDataset).MovieFilePaths.ElementAt((TestLoader as MovieFileDataLoader).MovieIndex[cell.Address.RowNumber - testRows.FirstOrDefault()].asscalar<int>());
            }
        }

        /// <summary>
        /// Writes the horizontal result to the record file.
        /// </summary>
        /// <param name="epochResult">The result of the epoch.</param>
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
        /// Calculates additional loss for the model.
        /// </summary>
        /// <param name="loss">The current loss value.</param>
        /// <returns>The additional loss value.</returns>
        public virtual Variable CalcAdditionalLoss(Variable loss)
        {
            return loss + L2Regularization.Invoke(Model.Params(), new NDarray(0.01).ToVariable())[0];
        }

        /// <summary>
        /// Calculates the loss.
        /// </summary>
        /// <remarks>
        /// <para>Types of loss functions:</para>
        /// <para> 1. Regression models:</para>
        /// <para>  - Mean Squared Error (MSE): Calculates the mean of the squared errors between the predicted values and the true values. It is one of the simplest loss functions and relatively inexpensive to compute. However, it is sensitive to outliers.</para>
        /// <para>  - Mean Absolute Error (MAE): Calculates the mean of the absolute errors between the predicted values and the true values. Unlike MSE, it is less sensitive to outliers.</para>
        /// <para>  - Huber Loss: A loss function that combines MSE and MAE. It behaves like MSE for small errors and like MAE for large errors. It is less sensitive to outliers and suitable for robust model training.</para>
        /// <para>  </para>
        /// <para> 2. Classification models:</para>
        /// <para>  - Cross-Entropy Loss: Calculates the information entropy between the predicted probabilities and the true labels. It is suitable for representing errors in probability distributions and is often used in multi-class classification.</para>
        /// <para>  - Hinge Loss: Calculates the magnitude of the error between the predicted values and the true labels. By introducing a misclassification penalty, it is suitable for training maximum margin classifiers like support vector machines.</para>
        /// <para>  - Log Loss: A loss function used in binary classification, such as logistic regression. It is similar to cross-entropy loss but slightly less expensive to compute.</para>
        /// <para>  </para>
        /// <para> 3. Generative models:</para>
        /// <para>  - Maximum Likelihood Estimation (MLE): Updates the parameters to maximize the probability of data generation. It is a simple method but can be computationally expensive.</para>
        /// <para>  - Evidence Lower Bound (ELBO): A method used to complement the shortcomings of MLE. By using integral calculations to approximate MLE, it can reduce computational costs.</para>
        /// <para>  - Adversarial Loss: A method that trains generative models by competing them against discriminative models. It has gained attention in recent years in fields such as image generation and natural language processing.</para>
        /// </remarks>
        /// <param name="y">Output from the model (predicted values)</param>
        /// <param name="t">Ground truth</param>
        /// <returns></returns>
        public virtual Variable CalcLoss(Variable y, NDarray t)
        {
            return SoftmaxCrossEntropy.Invoke(y, t.ToVariable())[0];
        }

        /// <summary>
        /// Calculates the evaluation metric
        /// </summary>
        /// <param name="y">Output from the model (predicted values)</param>
        /// <param name="t">Ground truth</param>
        /// <returns></returns>
        public virtual Variable CalcEvaluationMetric(Variable y, NDarray t)
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
            using var progress = _logger.BeginProgress($"{(Gpu.Available && EnableGpu ? "GPU Enabled" : "GPU Disabled")}...");
            progress.Complete();
        }
    }
}
