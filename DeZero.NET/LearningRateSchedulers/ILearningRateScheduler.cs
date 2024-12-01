namespace DeZero.NET.LearningRateSchedulers
{
    public interface ILearningRateScheduler
    {
        float GetLearningRate(int epoch, float currentLoss);
        void Reset();
    }
}
