using ClosedXML.Excel;
using DeZero.NET.Processes.CompletionHandler;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

namespace DeZero.NET.Processes
{
    /// <summary>
    /// Abstract class represents a parent process that starts a child process for training.
    /// </summary>
    public abstract class ParentProcess
    {
        private readonly IEnumerable<IProcessCompletionHandler> _completionHandlers;
        private Process CurrentProcess { get; set; }
        private int ProcessedEpoch { get; set; }

        /// <summary>
        /// Maximum number of epochs
        /// </summary>
        public int MaxEpoch { get; }

        /// <summary>
        /// Batch size
        /// </summary>
        public int BatchSize { get; }

        /// <summary>
        /// Whether to perform calculations using the GPU. If true, calculations are performed using the GPU; otherwise, they are not.
        /// </summary>
        public bool EnableGpu { get; }

        /// <summary>
        /// Path to the record file
        /// </summary>
        public abstract string RecordFilePath { get; }

        /// <summary>
        /// Path to the executable assembly of the child process
        /// </summary>
        public abstract string ExecutableAssembly { get; }

        /// <summary>
        /// Generates the arguments for the executable assembly of the child process.
        /// </summary>
        /// <param name="currentEpoch">The current epoch number.</param>
        /// <returns>A string representing the arguments for the executable assembly.</returns>
        public abstract string ExeArguments(int currentEpoch);

        /// <summary>
        /// Constructor for the parent process
        /// </summary>
        /// <param name="max_epoch">Maximum number of epochs</param>
        /// <param name="batch_size">Batch size</param>
        /// <param name="enableGpu">Whether to perform calculations using the GPU. If true, calculations are performed using the GPU; otherwise, they are not.</param>
        /// <param name="completionHandlers">Completion handlers to be executed after the training process is completed</param>
        protected ParentProcess(int max_epoch, int batch_size, bool enableGpu, IEnumerable<IProcessCompletionHandler> completionHandlers = null)
        {
            this.MaxEpoch = max_epoch;
            this.BatchSize = batch_size;
            this.EnableGpu = enableGpu;
            _completionHandlers = completionHandlers ?? Array.Empty<IProcessCompletionHandler>();

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
                        var firstRow_lastColumnCell = worksheet.Row(1).CellsUsed().LastOrDefault();

                        // 1列目の埋まっているセルのうち最も下のセルを取得
                        var firstColumn_lastRowCell = worksheet.Column(1).CellsUsed().LastOrDefault();

                        if (firstColumn_lastRowCell.GetString() == "s")
                        {
                            //Single Process per Epoch (Horizontal)

                            if (firstRow_lastColumnCell != null)
                            {
                                var lastCellValue = firstRow_lastColumnCell.GetValue<string>();

                                // セルの値が数字であれば変換して返す
                                if (int.TryParse(lastCellValue, out int currentEpoch))
                                {
                                    return currentEpoch;
                                }
                            }
                        }
                        else if (firstRow_lastColumnCell.GetString() == "s")
                        {
                            //Multiple Processes per Epoch (Vertical)

                            var fiveColumn_lastRowCell = worksheet.Column(5).CellsUsed().LastOrDefault();
                            var fiveColumn_lastRowCellRowNumber = fiveColumn_lastRowCell.Address.RowNumber;
                            var thirdColumn_detectLastRowCell = worksheet.Cell(fiveColumn_lastRowCellRowNumber, 3);
                            var thirdColumn_detectLastRowCellValue = thirdColumn_detectLastRowCell.GetString();

                            if (worksheet.Cell(fiveColumn_lastRowCellRowNumber, 2).GetString() == "epoch")
                            {
                                return 0;
                            }

                            var currentEpochStr = worksheet.Cell(fiveColumn_lastRowCellRowNumber, 2).GetString();
                            var currentEpoch = int.Parse(currentEpochStr);
                            if (thirdColumn_detectLastRowCellValue == "testtotal")
                            {
                                //既存のエポックは完了済み
                                return currentEpoch;
                            }
                            else
                            {
                                //既存のエポックは未完了
                                return currentEpoch - 1;
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
        /// Starts the training process. The training is executed in a child process.
        /// </summary>
        public virtual async void Fit()
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

            // Execute all completion handlers
            foreach (var handler in _completionHandlers)
            {
                await handler.OnProcessComplete("weights", RecordFilePath);
            }
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
                        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                        {
                            //一行上の行頭にカーソルを移動
                            Console.SetCursorPosition(0, Console.CursorTop - 1);
                        }
                        else
                        {
                            //一行上の行頭にカーソルを移動
                            Console.Write("\u001b[F");
                        }
                    }
                }
            };

            process.BeginOutputReadLine();
        }
    }
}
