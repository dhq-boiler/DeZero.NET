using System.Text;
using DeZero.NET.Core;
using DeZero.NET.matplotlib;

namespace DeZero.NET.Monitors
{
    public class RealtimeLossPlotter
    {
        private readonly int _maxEpoch;
        private readonly int _batchSize;
        private readonly Queue<Queue<float>> _losses;
        private readonly Queue<Queue<float>> _errors;
        private readonly Queue<Queue<float>> _learningRates;
        private readonly Queue<int> _frames;
        private readonly object _lockObject = new object();
        internal string[] Colors { get; private set; }

        public float CurrentLoss { get; private set; }

        public RealtimeLossPlotter(int maxEpoch, int batchSize)
        {
            _maxEpoch = maxEpoch;
            _batchSize = batchSize;
            _losses = new Queue<Queue<float>>();
            _errors = new Queue<Queue<float>>();
            _learningRates = new Queue<Queue<float>>();
            _frames = new Queue<int>();

            LoadColorPalette(Path.Combine("colors", "palette.txt"));

            // プロットの初期設定
            pyplot.figure();
            pyplot.title("Training Progress");
            pyplot.grid(true);
        }

        public void Update(int frame, float loss, float error, float learningRate, int epoch, bool render = true)
        {
            CurrentLoss = loss;
            
            lock (_lockObject)
            {
                Queue<float> losses = null;
                Queue<float> errors = null;
                Queue<float> learningRates = null;

                if (epoch <= _losses.Count)
                {
                    losses = _losses.ElementAt(epoch - 1);
                    errors = _errors.ElementAt(epoch - 1);
                    learningRates = _learningRates.ElementAt(epoch - 1);
                }
                else
                {
                    losses = new Queue<float>();
                    errors = new Queue<float>();
                    learningRates = new Queue<float>();
                    _losses.Enqueue(losses);
                    _errors.Enqueue(errors);
                    _learningRates.Enqueue(learningRates);
                }

                // データの追加
                losses.Enqueue(loss);
                errors.Enqueue(error);
                learningRates.Enqueue(learningRate);
                _frames.Enqueue(frame);

                if (!render) return;

                // プロットの更新
                pyplot.clf();

                double average = losses.Average();

                // Loss のプロット
                pyplot.subplot(311);  // 2行1列の1番目

                var line2dList = new List<(Line2D, int)>();

                foreach (var (l, c) in _losses.Select((x, i) => new { Index = i, Value = x }).Zip(Colors))
                {
                    if (l.Value.Count <= 1)
                    {
                        continue;
                    }

                    line2dList.Add((pyplot.plot(_frames.Take(l.Value.Count()).ToArray(), l.Value.Select(x => (double)x).ToArray(), "-", label: $"Epoch {l.Index + 1}", color: c)[0], l.Index + 1));
                }

                if (_frames.Count >= 2)
                {
                    pyplot.axhline(average, color: "r", linestyle: "-", alpha: 0.5d);
                    pyplot.text(_frames.Last(), average, $"avg.: {average:F2}", color: "r", va: "bottom", ha: "right",
                        fontsize: 8);
                }

                pyplot.ylabel("Loss (log scale)");
                pyplot.yscale("log");
                pyplot.grid(true);
                
                pyplot.legend(line2dList.Select(x => x.Item1).ToArray(), line2dList.Select(x => $"epoch {x.Item2}").ToArray());

                // Error のプロット
                pyplot.subplot(312);
                foreach (var (e, c) in _errors.Select((x, i) => new { Index = i, Value = x }).Zip(Colors))
                {
                    if (e.Value.Count <= 1)
                    {
                        continue;
                    }

                    line2dList.Add((pyplot.plot(_frames.Take(e.Value.Count()).ToArray(), e.Value.Select(x => (double)x).ToArray(), "-", label: $"Epoch {e.Index + 1}", color: c)[0], e.Index + 1));
                }

                pyplot.ylabel("Error");
                pyplot.yscale("log");
                pyplot.grid(true);

                //統計情報の表示
                pyplot.subplot(313);
                pyplot.axis("off");
                pyplot.text(0.5, 0.75, $"Loss: {loss:F6}", fontsize: 20, ha: "center", va: "center");
                pyplot.text(0.5, 0.50, $"Error: {error:F6}", fontsize: 20, ha: "center", va: "center");
                pyplot.text(0.5, 0.25, $"Learning Rate: {learningRate:F6}", fontsize: 20, ha: "center", va: "center");

                // グラフの更新
                pyplot.pause(0.001);
            }
        }

        public void LoadLoss(string filename, int currentEpoch)
        {
            try
            {
                for (int i = 1; i < currentEpoch; i++)
                {
                    var targetFilePath = Path.Combine("losses", $"{filename}_{i}.npy");

                    if (!File.Exists(targetFilePath))
                    {
                        continue;
                    }

                    var ndarray = Numpy.np.load(targetFilePath);
                    var losses = ndarray[0];
                    var errors = ndarray[1];
                    var learningRates = ndarray[2];
                    for (int j = 0; j < losses.len; j++)
                    {
                        Update(j * _batchSize, losses[j].asscalar<float>(), errors[j].asscalar<float>(), learningRates[j].asscalar<float>(),
                            i, render: j == losses.len - 1);
                    }
                }
            }
            finally
            {
            }
        }

        public void SaveLoss(string filename, int epoch)
        {
            if (!Directory.Exists("losses"))
            {
                Directory.CreateDirectory("losses");
            }

            try
            {
                using var loss = Numpy.np.array(_losses.ElementAt(epoch - 1).ToArray());
                using var error = Numpy.np.array(_errors.ElementAt(epoch - 1).ToArray());
                using var learningRate = Numpy.np.array(_learningRates.ElementAt(epoch - 1).ToArray());
                using var ndarray = Numpy.np.vstack(loss, error, learningRate);
                Numpy.np.save(Path.Combine("losses", Path.GetFileNameWithoutExtension(filename) + ".npy"), ndarray);
            }
            finally
            {
            }
        }

        public void LoadColorPalette(string filename)
        {
            if (!File.Exists(filename))
            {
                var colors = VividColors.GetUniqueRandomColors(_maxEpoch);
                SaveColorPalette(filename, colors);
            }

            Colors = File.ReadAllLines(filename);
        }

        public void SaveColorPalette(string filename, IReadOnlyList<string> colors)
        {
            if (!Directory.Exists("colors"))
            {
                Directory.CreateDirectory("colors");
            }

            var sb = new StringBuilder();
            foreach (var color in colors)
            {
                sb.AppendLine(color);
            }
            File.WriteAllText(filename, sb.ToString());
        }

        public void Clear()
        {
            _losses.Clear();
            _learningRates.Clear();
            _frames.Clear();
        }
    }
}
