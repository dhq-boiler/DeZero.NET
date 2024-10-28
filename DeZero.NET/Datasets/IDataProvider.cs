using System.Diagnostics;

namespace DeZero.NET.Datasets
{
    public interface IDataProvider : IEnumerable<(NDarray, NDarray)>
    {
        long Length { get; }

        Action<double, double, double, string, Stopwatch> OnSwitchDataFile { get; set; }

        void NotifyEvalValues(double loss, double error, double accuracy, Stopwatch sw);
    }
}
