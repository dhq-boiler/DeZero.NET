using System;
using System.Collections.Generic;
using System.Linq;

namespace DeZero.NET.LearningRateSchedulers
{
    public class ReduceLROnPlateau : ILearningRateScheduler
    {
        private readonly float _initialLr;
        private readonly float _minLr;
        private readonly float _maxLr;

        // 動画とエポックの設定
        //private const int TOTAL_VIDEOS = 19;
        //private const int TOTAL_EPOCHS = 20;
        //private const float TOTAL_STEPS = TOTAL_VIDEOS * TOTAL_EPOCHS;

        // ロスの閾値設定
        private const float HIGH_LOSS_THRESHOLD = 500.0f;
        private const float LOW_LOSS_THRESHOLD = 10.0f;

        // 基本的な減衰量
        private readonly float _decayPerStep;

        private float _currentLr;
        private int _currentStep;
        private readonly Queue<float> _recentLosses;
        private DateTime _lastAdjustment;

        public ReduceLROnPlateau(
            int totalStep,
            float initialLr = 0.1f,
            float minLr = 0.0001f,
            float maxLr = 0.1f)
        {
            _initialLr = initialLr;
            _currentLr = initialLr;
            _minLr = minLr;
            _maxLr = maxLr;

            // 全ステップでの基本減衰量を計算
            _decayPerStep = (initialLr - minLr) / totalStep;

            _currentStep = 0;
            _recentLosses = new Queue<float>();
            _lastAdjustment = DateTime.Now;
            Reset();
        }

        public float GetInitialLearningRate(int epoch)
        {
            float baseLr = _initialLr - (_decayPerStep * _currentStep);
            return _currentLr = Math.Max(baseLr, _minLr);
        }

        public float GetLearningRate(int epoch, float currentLoss)
        {
            if (float.IsNaN(currentLoss) || float.IsInfinity(currentLoss))
            {
                return _currentLr;
            }

            // 最小間隔（0.1秒）未満の場合は調整をスキップ
            var timeSinceLastAdjustment = (DateTime.Now - _lastAdjustment).TotalSeconds;
            if (timeSinceLastAdjustment < 0.1)
            {
                return _currentLr;
            }

            // 基本の線形減衰を適用
            float baseLr = _initialLr - (_decayPerStep * _currentStep);
            _currentLr = Math.Max(baseLr, _minLr);

            // ロスに基づく動的調整
            if (currentLoss > HIGH_LOSS_THRESHOLD)
            {
                // 高ロス時は学習率を一時的に下げる
                _currentLr *= 0.8f;
            }
            else if (currentLoss > LOW_LOSS_THRESHOLD)
            {
                // 中程度のロスの場合は現在の学習率を維持
                _currentLr = baseLr;
            }
            // 低ロスの場合は減衰された学習率をそのまま使用

            // 学習率の範囲を制限
            _currentLr = Math.Max(Math.Min(_currentLr, _maxLr), _minLr);

            _currentStep++;
            _lastAdjustment = DateTime.Now;

            UpdateLossHistory(currentLoss);

            return _currentLr;
        }

        private void UpdateLossHistory(float currentLoss)
        {
            _recentLosses.Enqueue(currentLoss);
            if (_recentLosses.Count > 5) // 直近5回分のロスを保持
            {
                _recentLosses.Dequeue();
            }
        }

        public void Reset()
        {
            _currentLr = _initialLr;
            _currentStep = 0;
            _recentLosses.Clear();
            _lastAdjustment = DateTime.Now;
        }
    }
}