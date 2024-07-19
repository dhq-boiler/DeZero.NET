using ClosedXML.Excel;
using System.Diagnostics;
using System.Text;

namespace DeZero.NET.Processes
{
    public abstract class ParentProcess
    {
        private Process CurrentProcess { get; set; }
        private int ProcessedEpoch { get; set; }

        public int MaxEpoch { get; }
        public int BatchSize { get; }
        public bool EnableGpu { get; }
        public abstract string RecordFilePath { get; }
        public abstract string ExecutableAssembly { get; }
        public abstract string ExeArguments(int currentEpoch);

        /// <summary>
        /// 親プロセスのコンストラクタ
        /// </summary>
        /// <param name="max_epoch">最大エポック数</param>
        /// <param name="batch_size">バッチサイズ</param>
        /// <param name="enableGpu">GPUによる計算を行うかどうか.trueならGPUによる計算を行う.そうでなければGPUによる計算を行わない.</param>
        protected ParentProcess(int max_epoch, int batch_size, bool enableGpu)
        {
            this.MaxEpoch = max_epoch;
            this.BatchSize = batch_size;
            this.EnableGpu = enableGpu;
            SetConsoleOutputEncoding();
            SetProcessExit();
            LoadCurrentEpochFromExcel();
        }

        private static void SetConsoleOutputEncoding()
        {
            Console.OutputEncoding = Encoding.UTF8;
        }

        private void LoadCurrentEpochFromExcel()
        {
            ProcessedEpoch = LoadFromExcel(RecordFilePath);
        }

        private void SetProcessExit()
        {
            AppDomain.CurrentDomain.ProcessExit += (sender, e) =>
            {
                if (CurrentProcess is not null)
                {
                    CurrentProcess.Kill();
                }
            };
        }

        private int LoadFromExcel(string mnistResultXlsx)
        {
            if (!System.IO.File.Exists(mnistResultXlsx))
            {
                return 0;
            }

            try
            {
                var count = 0;

                while (count < 100)
                {
                    try
                    {
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
                    }
                    catch (Exception e)
                    {
                        count++;
                        Thread.Sleep(1000);
                        continue;
                    }

                    break;
                }

                // セルが空または数字でない場合は0を返す
                return 0;
            }
            catch (Exception e)
            {
                Console.WriteLine($"The process cannot open the file '{mnistResultXlsx}' because it is being used by another process.");
                Console.WriteLine($"Try to sort out the processes you're running.");
                return int.MaxValue;
            }
        }

        /// <summary>
        /// トレーニングを開始します。トレーニングの処理は子プロセスで行われます。
        /// </summary>
        public virtual void Fit()
        {
            if (MaxEpoch - ProcessedEpoch <= 0)
            {
                Console.WriteLine("The training has already been completed.");
                return;
            }

            Console.WriteLine($"{DateTime.Now} Start training.");
            Console.WriteLine("==================================================================================");

            foreach (var epoch in Enumerable.Range(ProcessedEpoch, MaxEpoch - ProcessedEpoch))
            {
                StartProcessAndWait(ExecutableAssembly, ExeArguments(epoch + 1));
            }

            Console.WriteLine("==================================================================================");
            Console.WriteLine($"{DateTime.Now} Finish training.");
        }

        private void StartProcessAndWait(string filename, string arguments, string workingDir = null)
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

        private void RedirectStandardOutputToConsole(Process process)
        {
            process.OutputDataReceived += (sender, e) =>
            {
                if (!string.IsNullOrEmpty(e.Data))
                {
                    Console.CursorVisible = false;
                    Console.WriteLine(e.Data);
                    if (e.Data.EndsWith(" "))
                    {
                        Console.SetCursorPosition(0, Console.CursorTop - 1);
                    }
                }
            };

            process.BeginOutputReadLine();
        }
    }
}
