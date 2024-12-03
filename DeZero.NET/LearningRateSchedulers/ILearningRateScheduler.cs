namespace DeZero.NET.LearningRateSchedulers
{
    public interface ILearningRateScheduler
    {
        float GetInitialLearningRate(int epoch);
        float GetLearningRate(int epoch, float currentLoss);
        void Reset();
    }
}
