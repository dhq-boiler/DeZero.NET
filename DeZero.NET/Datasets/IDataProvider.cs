using DeZero.NET.Core;
using DeZero.NET.Processes;
using System.Diagnostics;

namespace DeZero.NET.Datasets
{
    public interface IDataProvider : IEnumerable<(NDarray, NDarray)>
    {
        long Length { get; }
        NDarray MovieIndex { get; set; }
        int CurrentMovieIndex { get; set; }
        long CurrentFrameIndex { get; }

        Action<ResultMetrics, string, Stopwatch> OnSwitchDataFile { get; set; }
        WorkerProcess WorkerProcess { get; set; }

        void SetResultMetricsAndStopwatch(ResultMetrics resultMetrics, Stopwatch sw);

        void SetLocalStopwatch(Stopwatch sw);
    }
}
