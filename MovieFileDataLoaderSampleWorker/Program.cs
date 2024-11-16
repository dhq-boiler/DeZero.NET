using DeZero.NET;
using DeZero.NET.Datasets;
using DeZero.NET.Extensions;
using DeZero.NET.Functions;
using DeZero.NET.Models;
using DeZero.NET.Optimizers;
using MHWGoldCrownModelTrainWorker;
using MovieFileDataLoaderSampleWorker;

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
workerProcess.SetModel(() => new DCNNModel(isVerbose: false, logLevel: DeZero.NET.Log.LogLevel.Error));
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
        DisposeAllInputs = true;
    }

    // 勾配クリッピングのための閾値
    private const float GRAD_CLIP_THRESHOLD = 1.0f;

    public override Variable CalcAdditionalLoss(Variable loss)
    {
        // L2正則化の強度を調整
        return loss + L2Regularization.Invoke(Model.Params(), new NDarray(0.001f).ToVariable())[0];
    }

    // 正規化された損失を計算する補助メソッド
    private Variable CalculateNormalizedLoss(Variable y, NDarray t)
    {
        try
        {
            var batch_size = y.Data.Value.shape[0];
            var feature_dim = y.Data.Value.shape[1];
            var normalized_losses = new List<Variable>();

            for (int dim = 0; dim < feature_dim; dim++)
            {
                // Variableとして平均と標準偏差を計算
                // 次元方向の平均
                var y_mean = DeZero.NET.Functions.Mean.Invoke(y, axis: 1, keepdims: true)[0];
                var y_std = DeZero.NET.Functions.Sqrt.Invoke(DeZero.NET.Functions.Mean.Invoke(DeZero.NET.Functions.Square.Invoke(y - y_mean)[0], axis: 1, keepdims: true)[0] + 1e-8f)[0];

                // 正規化 - すべてVariableの演算として実行
                var normalized_y = (y - y_mean) / y_std;

                // tをVariableに変換（一度だけ）
                var t_variable = t.ToVariable();
                var t_mean = DeZero.NET.Functions.Mean.Invoke(t_variable, axis: 1, keepdims: true)[0];
                var t_std = DeZero.NET.Functions.Sqrt.Invoke(DeZero.NET.Functions.Mean.Invoke(DeZero.NET.Functions.Square.Invoke(t_variable - t_mean)[0], axis: 1, keepdims: true)[0] + 1e-8f)[0];
                var normalized_t = (t_variable - t_mean) / t_std;

                // Huber Lossの計算
                var diff = normalized_y - normalized_t;
                var abs_diff = DeZero.NET.Functions.Abs.Invoke(diff)[0];
                var delta = DeZero.NET.Functions.Const.Invoke(1.0f)[0];

                // Huber Loss - すべてFunctionsを使用
                var quadratic = DeZero.NET.Functions.Mul.Invoke(Const.Invoke(0.5f)[0], DeZero.NET.Functions.Square.Invoke(diff)[0])[0];
                var linear = DeZero.NET.Functions.Sub.Invoke(
                DeZero.NET.Functions.Mul.Invoke(delta[0], abs_diff[0])[0],
                    DeZero.NET.Functions.Mul.Invoke(DeZero.NET.Functions.Const.Invoke(0.5f)[0], DeZero.NET.Functions.Square.Invoke(delta[0])[0])[0]
                )[0];

                var condition = DeZero.NET.Functions.LessThan.Invoke(abs_diff, delta).Item1[0];
                var huber_loss = DeZero.NET.Functions.Where.Invoke(condition, quadratic, linear).Item1[0];

                // スケーリング係数もVariableとして計算
                var scaling_factor = DeZero.NET.Functions.Div.Invoke(t_std, y_std)[0];
                var scaled_loss = DeZero.NET.Functions.Mul.Invoke(huber_loss, scaling_factor)[0];

                normalized_losses.Add(scaled_loss);
            }

            // 損失の結合
            var combined_loss = normalized_losses[0];
            for (int i = 1; i < normalized_losses.Count; i++)
            {
                combined_loss = DeZero.NET.Functions.Add.Invoke(combined_loss, normalized_losses[i]).Item1[0];
            }

            // バッチ全体の平均
            var total_loss = DeZero.NET.Functions.Sum.Invoke(combined_loss)[0];
            var mean_loss = DeZero.NET.Functions.Div.Invoke(total_loss, Const.Invoke(batch_size * feature_dim)[0])[0];

            return mean_loss;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error in normalized loss calculation: {ex.Message}");
            throw;
        }
    }

    public override Variable CalcLoss(Variable y, NDarray t)
    {
        try
        {
            var loss = CalculateNormalizedLoss(y, t);

            // トレーニングフェーズでのみ勾配クリッピングを試みる
            if (Config.EnableBackprop)
            {
                // 一時的な変数を作成して勾配を計算
                using var temp_loss = loss.Data.Value.copy().ToVariable();
                temp_loss.Backward(retain_grad: true);

                // 勾配が存在する場合のみクリッピングを適用
                if (temp_loss.Grad != null)
                {
                    var grad_norm = xp.linalg.norm(temp_loss.Grad.Value.Data.Value);
                    if (grad_norm.asscalar<float>() != float.NaN && grad_norm.asscalar<float>() > GRAD_CLIP_THRESHOLD)
                    {
                        var scale = GRAD_CLIP_THRESHOLD / (grad_norm.asscalar<float>() + 1e-8f);
                        loss = new Variable(loss.Data.Value);
                        loss.Grad.Value = new Variable(temp_loss.Grad.Value.Data.Value * scale);
                    }
                }
            }

            return loss;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error in loss calculation: {ex.Message}");
            throw;
        }
    }

    public override Variable CalcEvaluationMetric(Variable y, NDarray t)
    {
        // 各次元の平均絶対誤差を計算
        var batch_size = y.Data.Value.shape[0];
        var feature_dim = y.Data.Value.shape[1];

        var diff = y - t.ToVariable();
        var abs_diff = Abs.Invoke(diff)[0];

        // すべての次元の誤差を合計し、サンプル数と次元数で割って平均を取る
        return Sum.Invoke(abs_diff)[0] / (batch_size * feature_dim);
    }

    protected override Func<NDarray, long> UnitLength => (t) => 1;

    public override ModelType ModelType => ModelType.Regression;
}
