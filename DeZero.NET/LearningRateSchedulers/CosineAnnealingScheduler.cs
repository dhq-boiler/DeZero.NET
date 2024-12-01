namespace DeZero.NET.LearningRateSchedulers
{
    public class CosineAnnealingScheduler : ILearningRateScheduler
    {
        private readonly float _initialLr;
        private readonly float _minLr;
        private readonly int _totalEpochs;
        private readonly float _warmupEpochs;

        public CosineAnnealingScheduler(
            float initialLr = 0.001f,
            float minLr = 0.00001f,
            int totalEpochs = 100,
            float warmupEpochs = 5)
        {
            _initialLr = initialLr;
            _minLr = minLr;
            _totalEpochs = totalEpochs;
            _warmupEpochs = warmupEpochs;
        }

        public float GetLearningRate(int epoch, float currentLoss)
        {
            // Warmup period
            if (epoch < _warmupEpochs)
            {
                return _initialLr * ((float)epoch / _warmupEpochs);
            }

            // Cosine annealing
            float progress = (epoch - _warmupEpochs) / (float)(_totalEpochs - _warmupEpochs);
            progress = Math.Min(1.0f, Math.Max(0.0f, progress));

            float cosine = (float)(0.5 * (1 + Math.Cos(Math.PI * progress)));
            return _minLr + (_initialLr - _minLr) * cosine;
        }

        public void Reset()
        {
            // Reset any internal state if needed
        }
    }
}
