using DeZero.NET.Log;

namespace DeZero.NET.LearningRateSchedulers
{
    public class LearningRateManager
    {
        private readonly ILearningRateScheduler _scheduler;
        private readonly ILogger _logger;
        private float _currentLr;

        public LearningRateManager(
            ILearningRateScheduler scheduler,
            float initialLr = 0.001f,
            ILogger logger = null)
        {
            _scheduler = scheduler;
            _currentLr = initialLr;
            _logger = logger ?? new ConsoleLogger(LogLevel.Info, false);
        }

        public float GetInitialLearningRate(int epoch)
        {
            return _scheduler.GetInitialLearningRate(epoch);
        }

        public float UpdateLearningRate(int epoch, float currentLoss)
        {
            float newLr = _scheduler.GetLearningRate(epoch, currentLoss);

            if (Math.Abs(newLr - _currentLr) > float.Epsilon)
            {
                _logger.LogInfo($"Epoch {epoch}: Adjusting learning rate from {_currentLr:F6} to {newLr:F6}");
                _currentLr = newLr;
            }

            return _currentLr;
        }

        public void Reset()
        {
            _scheduler.Reset();
        }
    }
}
