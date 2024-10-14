var max_epoch = 50;
var batch_size = 100;
var hidden_size = 1000;
var enableGpu = true;

var parentProcess = new ParentProcess(max_epoch, batch_size, enableGpu);

parentProcess.Fit();

class ParentProcess : DeZero.NET.Processes.ParentProcess
{
    public override string RecordFilePath => "MovieFileDataLoaderSample_result.xlsx";
    public override string ExecutableAssembly => "MovieFileDataLoaderSample.exe";

    public override string ExeArguments(int currentEpoch)
    {
        return $"{currentEpoch} {BatchSize} {1000} {EnableGpu} '{RecordFilePath}'";
    }

    public ParentProcess(int max_epoch, int batch_size, bool enableGpu) : base(max_epoch, batch_size, enableGpu)
    {
    }
}