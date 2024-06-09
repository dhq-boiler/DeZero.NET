using System.Diagnostics;
using ClosedXML.Excel;

var max_epoch = 100;
var batch_size = 100;
var hidden_size = 1000;
var enableGpu = true;
const string xlsx_path = "MNIST_result.xlsx";

Process CurrentProcess = default;

AppDomain.CurrentDomain.ProcessExit += (sender, e) =>
{
    if (CurrentProcess is not null)
    {
        CurrentProcess.Kill();
    }
};

var processedEpoch = LoadFromExcel(xlsx_path);

int LoadFromExcel(string mnistResultXlsx)
{
    if (!System.IO.File.Exists(mnistResultXlsx))
    {
        return 0;
    }
    using var workbook = new XLWorkbook(mnistResultXlsx);
    var worksheet = workbook.Worksheet(1);

    // 1行目の埋まっているセルのうち最も右のセルを取得
    var lastCell = worksheet.Row(1).CellsUsed().LastOrDefault();

    if (lastCell != null)
    {
        var lastCellValue = lastCell.GetValue<string>();

        // セルの値が数字であれば変換して返す
        if (int.TryParse(lastCellValue, out int currentEpoch))
        {
            return currentEpoch;
        }
    }

    // セルが空または数字でない場合は0を返す
    return 0;
}

foreach (var epoch in Enumerable.Range(processedEpoch, max_epoch))
{
    StartProcessAndWait("DeZero.NET.MNISTSampleWorker.exe", $"{epoch + 1} {batch_size} {hidden_size} {enableGpu}");
}

Console.WriteLine("==================================================================================");
Console.WriteLine($"{DateTime.Now} Finish training.");

void StartProcessAndWait(string filename, string arguments, string workingDir = null)
{
    var psi = new ProcessStartInfo(filename)
    {
        UseShellExecute = false,
        RedirectStandardOutput = true,
        RedirectStandardError = true,
        CreateNoWindow = true,
        WorkingDirectory = workingDir,
        Arguments = arguments,
    };
    Console.WriteLine($"{DateTime.Now} {(workingDir is null ? Directory.GetCurrentDirectory() : workingDir)}> {filename} {arguments}");
    CurrentProcess = Process.Start(psi);
    RedirectStandardOutputToConsole(CurrentProcess);
    while (!File.Exists("signal"))
    {
        Thread.Sleep(1000);
    }
    File.Delete("signal");
    CurrentProcess.Kill();
    CurrentProcess.CancelOutputRead();
    CurrentProcess = null;
}

void RedirectStandardOutputToConsole(Process process)
{
    process.OutputDataReceived += (sender, e) =>
    {
        if (!string.IsNullOrEmpty(e.Data))
        {
            Console.WriteLine(e.Data);
        }
    };

    process.BeginOutputReadLine();
}