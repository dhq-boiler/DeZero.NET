using DeZero.NET.Log;
using DeZero.NET.Processes.CompletionHandler;

var minimumLogLevel = LogLevel.Info;
var isVerbose = false;

var max_epoch = 5;
var batch_size = 16;
var enableGpu = true;

var parentProcess = new ParentProcess(max_epoch, batch_size, enableGpu, new ConsoleLogger(minimumLogLevel, isVerbose));

parentProcess.Fit();

class ParentProcess : DeZero.NET.Processes.ParentProcess
{
    public ParentProcess(int max_epoch, int batch_size, bool enableGpu, ILogger logger, IEnumerable<IProcessCompletionHandler> completionHandlers = null)
        : base(max_epoch, batch_size, enableGpu, logger, completionHandlers)
    {
    }

    public override string RecordFilePath => "MovieFileDataLoaderSample_result.xlsx";
    public override string ExecutableAssembly => "MovieFileDataLoaderSampleWorker.exe";

    public override string ExeArguments(int currentEpoch)
    {
        return $"{currentEpoch} {BatchSize} {1000} {EnableGpu} '{RecordFilePath}'";
    }
}