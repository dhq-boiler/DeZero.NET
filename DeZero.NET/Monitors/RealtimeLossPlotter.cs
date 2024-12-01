using DeZero.NET.matplotlib;

namespace DeZero.NET.Monitors
{
    public class RealtimeLossPlotter
    {
        private readonly Queue<float> _losses;
        private readonly Queue<float> _learningRates;
        private readonly Queue<int> _frames;
        private readonly int _maxPoints;
        private readonly object _lockObject = new object();

        public float CurrentLoss { get; private set; }

        public RealtimeLossPlotter(int maxPoints = 100)
        {
            _maxPoints = maxPoints;
            _losses = new Queue<float>(maxPoints);
            _learningRates = new Queue<float>(maxPoints);
            _frames = new Queue<int>(maxPoints);

            // プロットの初期設定
            pyplot.figure();
            pyplot.title("Training Progress");
            pyplot.grid(true);
        }

        public void Update(int frame, float loss, float learningRate)
        {
            CurrentLoss = loss;
            
            lock (_lockObject)
            {
                // データの追加
                _frames.Enqueue(frame);
                _losses.Enqueue(loss);
                _learningRates.Enqueue(learningRate);

                // キューのサイズ管理
                if (_frames.Count > _maxPoints)
                {
                    _frames.Dequeue();
                    _losses.Dequeue();
                    _learningRates.Dequeue();
                }

                // プロットの更新
                pyplot.clf();

                double average = _losses.Average();

                // Loss のプロット
                pyplot.subplot(211);  // 2行1列の1番目
                pyplot.plot(_frames.ToArray(), _losses.Select(x => (double)x).ToArray(), "b-", label: "Loss");

                if (_frames.Count >= 2)
                {
                    pyplot.axhline(average, color: "r", linestyle: "-", alpha: 0.5d);
                    pyplot.text(_frames.Last(), average, $"avg.: {average:F2}", color: "r", va: "bottom", ha: "right",
                        fontsize: 8);
                }

                pyplot.ylabel("Loss (log scale)");
                pyplot.yscale("log");
                pyplot.grid(true);
                pyplot.legend();

                pyplot.subplot(212);
                pyplot.axis("off");
                pyplot.text(0.5, 0.5, $"Loss: {loss:F6}", fontsize: 20, ha: "center", va: "center");
                pyplot.text(0.5, 0.25, $"Learning Rate: {learningRate:F6}", fontsize: 20, ha: "center", va: "center");

                // グラフの更新
                pyplot.pause(0.001);
            }
        }

        public void Clear()
        {
            _losses.Clear();
            _learningRates.Clear();
            _frames.Clear();
        }
    }
}
