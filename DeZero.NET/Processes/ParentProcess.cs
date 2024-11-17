using Amazon.EC2.Model;
using Amazon.EC2;
using ClosedXML.Excel;
using DeZero.NET.Processes.CompletionHandler;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using DeZero.NET.Log;

namespace DeZero.NET.Processes
{
    /// <summary>
    /// Abstract class represents a parent process that starts a child process for training.
    /// </summary>
    public abstract class ParentProcess
    {
        private const string CURSOR_UP = "__CURSOR_UP__";
        private const string PROGRESS_START = "__PROGRESS_START__";
        private const string PROGRESS_END = "__PROGRESS_END__";

        private readonly ILogger _logger;
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
        protected ParentProcess(int max_epoch, int batch_size, bool enableGpu, ILogger logger, IEnumerable<IProcessCompletionHandler> completionHandlers = null)
        {
            MaxEpoch = max_epoch;
            BatchSize = batch_size;
            EnableGpu = enableGpu;
            _logger = logger;
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
                _logger.LogInfo("The training has already been completed.");
                return;
            }

            _logger.LogInfo($"Start training.");
            _logger.LogInfo("==================================================================================");

            try
            {
                foreach (var epoch in Enumerable.Range(ProcessedEpoch, MaxEpoch - ProcessedEpoch))
                {
                    await StartProcessAndWaitAsync(ExecutableAssembly, ExeArguments(epoch + 1));
                }

                _logger.LogInfo("==================================================================================");
                _logger.LogInfo("Training completed successfully");

                // Execute all completion handlers
                foreach (var handler in _completionHandlers)
                {
                    await handler.OnProcessComplete("weights", RecordFilePath);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError($"Training failed: {ex.Message}");
                throw;
            }
        }

        private async Task StartProcessAndWaitAsync(string filename, string arguments, string workingDir = null)
        {
            var psi = new ProcessStartInfo(filename)
            {
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
                WorkingDirectory = workingDir,
                Arguments = arguments,
                // デバッグメッセージを抑制
                EnvironmentVariables =
            {
                ["DOTNET_CLI_UI_LANGUAGE"] = "en-US",
                ["COMPlus_EnableDiagnostics"] = "0"
            }
            };

            //using var progress = _logger.BeginProgress(
            //    $"{(workingDir is null ? Directory.GetCurrentDirectory() : workingDir)}> {filename} {arguments}");

            var timestamp = DateTime.Now.ToString("yyyy-MM-dd(ddd) HH:mm:ss.fff");
            var newMessage = $"{timestamp} [INFO] {(workingDir is null ? Directory.GetCurrentDirectory() : workingDir)}> {filename} {arguments}";
            Console.WriteLine(newMessage);

            try
            {
                CurrentProcess = Process.Start(psi);

                // エラー出力のリダイレクト
                CurrentProcess.ErrorDataReceived += (sender, e) =>
                {
                    if (!string.IsNullOrEmpty(e.Data) && !IsDebuggerMessage(e.Data))
                    {
                        _logger.LogError(e.Data);
                    }
                };
                CurrentProcess.BeginErrorReadLine();

                RedirectStandardOutputToLogger(CurrentProcess);

                // プロセスの終了とsignalファイルの両方を監視
                while (!CurrentProcess.HasExited)
                {
                    if (File.Exists("signal"))
                    {
                        break;
                    }

                    // より短い間隔で確認
                    Thread.Sleep(100);
                }

                File.Delete("signal");

                // プロセスの終了を適切に処理
                if (!CurrentProcess.HasExited)
                {
                    // 標準出力の読み取りを停止
                    CurrentProcess.CancelOutputRead();

                    // プロセスを終了
                    CurrentProcess.Kill();

                    // プロセスが確実に終了するまで待機
                    CurrentProcess.WaitForExit();
                }

                CurrentProcess.Dispose();
                CurrentProcess = null;

                //progress.Complete();
            }
            catch (Exception ex)
            {
                if (CurrentProcess != null)
                {
                    try
                    {
                        if (!CurrentProcess.HasExited)
                        {
                            CurrentProcess.CancelOutputRead();
                            CurrentProcess.Kill();
                            CurrentProcess.WaitForExit();
                        }
                        CurrentProcess.Dispose();
                        CurrentProcess = null;
                    }
                    catch
                    {
                        // プロセスのクリーンアップ中のエラーは無視
                    }
                }
                Console.WriteLine($"Process execution failed: {ex.Message}");
                //progress.Failed($"Process execution failed: {ex.Message}");
                throw;
            }
        }

        // デバッガーメッセージをフィルタリング
        private bool IsDebuggerMessage(string message)
        {
            return message.Contains("プロセス") && message.Contains("終了しました") ||
                   message.Contains("デバッグ") ||
                   message.Contains("このウィンドウを閉じるには");
        }

        private int stackLineCount = 0;
        private int beginLineWidth = 0;

        private string previousLine = string.Empty;

        private void RedirectStandardOutputToLogger(Process process)
        {
            var isProgressLine = false;
            var currentProgressLine = string.Empty;

            process.OutputDataReceived += async (sender, e) =>
            {
                if (string.IsNullOrEmpty(e.Data)) return;

                // 制御シーケンスの処理
                /*if (e.Data.Contains(CURSOR_UP))
                {
                    if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                    {
                        Console.SetCursorPosition(0, Console.CursorTop - 1);
                    }
                    else
                    {
                        Console.Write("\u001b[F");
                    }
                    stackLineCount = 0;
                    return;
                }
                else */if (previousLine.Contains("%"))
                {
                    if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                    {
                        Console.SetCursorPosition(0, Console.CursorTop - 1);
                    }
                    else
                    {
                        Console.Write("\u001b[F");
                    }
                    stackLineCount = 0;
                    if (e.Data.Contains(CURSOR_UP))
                    {
                        return;
                    }
                }
                else if (e.Data.Contains(PROGRESS_START))
                {
                    isProgressLine = true;
                    currentProgressLine = e.Data.Replace(PROGRESS_START, "");
                    if (currentProgressLine.Contains(PROGRESS_END))
                    {
                        currentProgressLine = currentProgressLine.Replace(PROGRESS_END, "");
                        beginLineWidth = currentProgressLine.Length;
                    }
                    Console.WriteLine(currentProgressLine);
                    stackLineCount = 0;
                    return;
                }
                else if (e.Data.Contains(PROGRESS_END))
                {
                    isProgressLine = false;
                    var willWriteLineNumber = Console.CursorTop - stackLineCount;
                    if (willWriteLineNumber < 0)
                    {
                        return;
                    }
                    Console.SetCursorPosition(beginLineWidth, willWriteLineNumber);
                    Console.WriteLine(e.Data.Replace(PROGRESS_END, ""));
                    return;
                }

                // 通常のログ出力かプログレス更新
                if (isProgressLine)
                {
                    stackLineCount++;
                    var newLine = $"\r{e.Data}";
                    Console.WriteLine(newLine);
                    previousLine = newLine;
                }
                else
                {
                    if (!IsDebuggerMessage(e.Data))
                    {
                        _logger.LogInfo(e.Data);
                    }

                    if (e.Data.EndsWith("__SHUTDOWN__"))
                    {
                        await HandleShutdownRequest();
                    }
                }
            };

            process.BeginOutputReadLine();
        }

        private async Task HandleShutdownRequest()
        {
            if (await IsRunningOnEC2() == false)
            {
                _logger.LogWarning("The process is not running on EC2. The instance will not be shut down.");
                return;
            }

            using var progress = _logger.BeginProgress("Processing shutdown request");
            try
            {
                string instanceId = Amazon.Util.EC2InstanceMetadata.InstanceId;
                await StopInstanceAsync(instanceId);
                progress.Complete("Instance shutdown initiated");
            }
            catch (Exception ex)
            {
                progress.Failed($"Shutdown failed: {ex.Message}");
            }
        }

        private async Task StopInstanceAsync(string instanceId)
        {
            try
            {
                var request = new StopInstancesRequest
                {
                    InstanceIds = new List<string> { instanceId }
                };

                var response = await new AmazonEC2Client().StopInstancesAsync(request);

                foreach (var instanceStateChange in response.StoppingInstances)
                {
                    Console.WriteLine($"Instance {instanceStateChange.InstanceId} is {instanceStateChange.CurrentState.Name}");
                }
            }
            catch (AmazonEC2Exception ex)
            {
                Console.WriteLine($"Error stopping instance: {ex.Message}");
                throw;
            }
        }

        private const string EC2TokenEndpoint = "http://169.254.169.254/latest/api/token";
        private const string EC2MetadataEndpoint = "http://169.254.169.254/latest/meta-data/";
        private const int TimeoutMilliseconds = 2000;

        public static async Task<bool> IsRunningOnEC2()
        {
            try
            {
                using (var httpClient = new HttpClient())
                {
                    httpClient.Timeout = TimeSpan.FromMilliseconds(TimeoutMilliseconds);

                    // IMDSv2トークンの取得
                    var tokenRequest = new HttpRequestMessage(HttpMethod.Put, EC2TokenEndpoint);
                    tokenRequest.Headers.Add("X-aws-ec2-metadata-token-ttl-seconds", "21600");
                    var tokenResponse = await httpClient.SendAsync(tokenRequest);

                    if (!tokenResponse.IsSuccessStatusCode)
                        return false;

                    var token = await tokenResponse.Content.ReadAsStringAsync();

                    // メタデータへのアクセス
                    var metadataRequest = new HttpRequestMessage(HttpMethod.Get, EC2MetadataEndpoint);
                    metadataRequest.Headers.Add("X-aws-ec2-metadata-token", token);
                    var response = await httpClient.SendAsync(metadataRequest);

                    return response.IsSuccessStatusCode;
                }
            }
            catch (Exception)
            {
                return false;
            }
        }
    }
}
