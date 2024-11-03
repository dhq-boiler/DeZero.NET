﻿using DeZero.NET.Processes.CompletionHandler;

var max_epoch = 5;
var batch_size = 32;
var enableGpu = true;

var parentProcess = new ParentProcess(max_epoch, batch_size, enableGpu);

parentProcess.Fit();

class ParentProcess : DeZero.NET.Processes.ParentProcess
{
    public ParentProcess(int max_epoch, int batch_size, bool enableGpu, IEnumerable<IProcessCompletionHandler> completionHandlers = null)
        : base(max_epoch, batch_size, enableGpu, completionHandlers)
    {
    }

    public override string RecordFilePath => "MovieFileDataLoaderSample_result.xlsx";
    public override string ExecutableAssembly => "MovieFileDataLoaderSampleWorker.exe";

    public override string ExeArguments(int currentEpoch)
    {
        return $"{currentEpoch} {BatchSize} {1000} {EnableGpu} '{RecordFilePath}'";
    }
}