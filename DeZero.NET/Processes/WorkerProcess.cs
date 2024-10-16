using ClosedXML.Excel;
using DeZero.NET.Core;
using DeZero.NET.Extensions;
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
        private void SaveOptimizer()
        {
            Console.Write($"{DateTime.Now} Save optimizer states...");
            Optimizer.SaveParameters();
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

        /// <summary>
        /// トレーニングとテストを実行します.
        /// </summary>
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
                sum_loss += total_loss.Data.Value.asscalar<float>() * t.len;
                sum_acc += acc.Data.Value.asscalar<float>() * t.len;
                count++;
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
                    if (DisposeAllInputs)
                    {
                        Model.DisposeAllInputs();
                    }
                    using var loss = CalcLoss(y, t);
                    var accuracy = new Accuracy();
                    using var acc = accuracy.Call(Params.New.SetKeywordArg(y, t))[0];
                    test_loss += loss.Data.Value.asscalar<float>() * t.len;
                    test_acc += acc.Data.Value.asscalar<float>() * t.len;
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
