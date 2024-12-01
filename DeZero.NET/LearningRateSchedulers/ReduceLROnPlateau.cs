namespace DeZero.NET.LearningRateSchedulers
{
    public class ReduceLROnPlateau : ILearningRateScheduler
    {
        private readonly float _factor;
        private readonly int _patience;
        private readonly float _minLr;
        private float _bestLoss;
        private int _waitCount;
        private float _currentLr;

        private readonly float _threshold = 0.01f;

        public ReduceLROnPlateau(
            float initialLr = 0.001f,
            float factor = 0.1f,
            int patience = 10,
            float minLr = 0.00001f)
        {
            _currentLr = initialLr;
            _factor = factor;
            _patience = patience;
            _minLr = minLr;
            Reset();
        }

        public float GetLearningRate(int epoch, float currentLoss)
        {
            // 相対的な改善を計算
            float improvement = (_bestLoss - currentLoss) / _bestLoss;

            if (improvement > _threshold)  // 閾値以上の改善があった場合
            {
                _bestLoss = currentLoss;
                _waitCount = 0;
            }
            else
            {
                _waitCount++;
                if (_waitCount >= _patience)
                {
                    if (_currentLr > _minLr)
                    {
                        _currentLr = Math.Max(_currentLr * _factor, _minLr);
                        _waitCount = 0;  // カウントをリセット
                    }
                }
            }

            return _currentLr;
        }

        public void Reset()
        {
            _bestLoss = float.MaxValue;
            _waitCount = 0;
        }
    }
}
