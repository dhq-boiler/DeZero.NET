namespace DeZero.NET.Core
{
    public class ResultMetrics
    {
        public double SumLoss { get; set; }
        public double SumError { get; set; }
        public double SumAccuracy { get; set; }

        public void Initialize()
        {
            SumLoss = 0;
            SumError = 0;
            SumAccuracy = 0;
        }
    }
}
