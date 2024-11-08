using DeZero.NET.OpenCv;
using System.Collections;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

namespace DeZero.NET.Datasets
{
    public class MovieFileDataLoader : IDataProvider
    {
        public MovieFileDataset Dataset { get; }
        public bool Shuffle { get; }
        public double MaxIter { get; }
        public int Iteration { get; protected set; }
        public int BatchSize { get; } = -1;
        public NDarray MovieIndex { get; set; }

        public int CurrentMovieIndex { get; set; }

        public long CurrentFrameIndex { get; protected set; }

        public Action ChangeMovieAction { get; set; }

        private VideoCapture VideoCapture { get; set; }

        public long Length => _FrameCount;
        public Action<double, double, double, string, Stopwatch> OnSwitchDataFile { get; set; }
        private double Loss { get; set; }
        private double Error { get; set; }
        private double Accuracy { get; set; }
        private Stopwatch Stopwatch { get; set; }

        public MovieFileDataLoader(MovieFileDataset dataset, int batchSize, Action changeMovieAction, bool shuffle = true, Action<double, double, double, string, Stopwatch> onSwitchDataFile = null)
        {
            Dataset = dataset;
            BatchSize = batchSize;
            Shuffle = shuffle;
            MaxIter = 1;
            ChangeMovieAction = changeMovieAction;
            OnSwitchDataFile = onSwitchDataFile;
            Reset();
        }

        protected void Reset()
        {
            Iteration = 0;
            CurrentMovieIndex = 0;
            MovieIndex?.Dispose();
            if (Shuffle)
            {
                MovieIndex = xp.random.permutation(Dataset.MovieFilePaths.Length);
            }
            else
            {
                MovieIndex = xp.arange(Dataset.MovieFilePaths.Length);
            }
        }

        private long _FrameCount = long.MaxValue;
        private Queue<(NDarray, NDarray)> _buffer = new Queue<(NDarray, NDarray)>();

        public virtual (IterationStatus, (NDarray[], NDarray[])) Next()
        {
            var frames = new List<NDarray>(BatchSize);
            var labels = new List<NDarray>(BatchSize);

            var ret = IterationStatus.Continue;

            while (frames.Count < BatchSize)
            {
                if (ret != IterationStatus.ChangeSource && _buffer.Count == 0)
                {
                    var status = FillBuffer();
                    if (status == IterationStatus.Break)
                    {
                        // バッファを埋められなかった場合（データセットの終わり）
                        if (frames.Count > 0)
                        {
                            return (IterationStatus.Break, (frames.ToArray(), labels.ToArray()));
                        }
                        return (IterationStatus.Break, (null, null));
                    }
                    if (_buffer.Count == 0)
                    {
                        // バッファを埋められなかった場合（データセットの終わり）
                        return (IterationStatus.Break, (null, null));
                    }
                    ret = status;
                }

                if (_buffer.Count == 0) break;

                var (frame, label) = _buffer.Dequeue();
                frames.Add(frame);
                labels.Add(label);
            }

            return (ret, (frames.ToArray(), labels.ToArray()));
        }

        private IterationStatus FillBuffer()
        {
            Gpu.Use = false;
            var ret = IterationStatus.Continue;
            if (CurrentFrameIndex == 0)
            {
                if (CurrentMovieIndex >= Dataset.MovieFilePaths.Length || MovieIndex.len - 1 < CurrentMovieIndex)
                {
                    return IterationStatus.Break;
                }

                int movieIndex = MovieIndex[CurrentMovieIndex].asscalar<int>();
                var targetFilePath = Dataset.MovieFilePaths[movieIndex];
                VideoCapture?.Dispose();
                VideoCapture = new VideoCapture(targetFilePath);

                if (!VideoCapture.IsOpened())
                {
                    throw new Exception("Movie file not found.");
                }

                _FrameCount = (long)VideoCapture.Get(VideoCaptureProperties.FrameCount);
                ConsoleOut();
                if (IsRunningFromVisualStudio())
                {
                    Console.SetCursorPosition(0, Console.CursorTop - 1);
                }
                else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                {
                    //一行上の行頭にカーソルを移動
                    Console.Write("\u001b[F");
                }

                ret = IterationStatus.ChangeSource;
            }

            while (_buffer.Count < BatchSize)
            {
                int movieIndex = 0;
                ret = CheckContinue(ref movieIndex);
                if (ret == IterationStatus.Break)
                {
                    return IterationStatus.Break;
                }

                VideoCapture.Set(VideoCaptureProperties.PosFrames, CurrentFrameIndex);
                VideoCapture.Retrieve(out var ndArray);

                movieIndex = MovieIndex[CurrentMovieIndex].asscalar<int>();

                if (Dataset.LabelArray[movieIndex].len <= CurrentFrameIndex)
                {
                    return ret;
                }

                var labelNdArray = Dataset.LabelArray[movieIndex][(int)CurrentFrameIndex];

                ndArray = ProcessFrame(ndArray);

                _buffer.Enqueue((ndArray, labelNdArray));

                CurrentFrameIndex++;
            }

            return ret;
        }

        public virtual NDarray ProcessFrame(NDarray ndArray)
        {
            return ndArray;
        }

        private IterationStatus CheckContinue(ref int movieIndex)
        {
            if (CurrentFrameIndex >= _FrameCount || CurrentFrameIndex >= Dataset.LabelArray[movieIndex].len)
            {
                if (CurrentMovieIndex >= MovieIndex.len)
                {
                    return IterationStatus.Break;
                }
                else
                {
                    return IterationStatus.ChangeSource;
                }
            }
            return IterationStatus.Continue;
        }

        public IEnumerator<(NDarray, NDarray)> GetEnumerator()
        {
            if (IsRunningFromVisualStudio())
            {
                Console.CursorVisible = true;
            }
            while (true)
            {
                Gpu.Use = false;
                var next = Next();
                var x = next.Item2.Item1;
                var t = next.Item2.Item2;

                if (next.Item1 == IterationStatus.ChangeSource)
                {
                    CurrentFrameIndex = _FrameCount;

                    Gpu.Use = true;
                    yield return (xp.array(x.Select(y => y).ToArray()), xp.array(t.Select(y => y).ToArray()));
                    Gpu.Use = false;

                    ConsoleOut();

                    CurrentFrameIndex = 0;
                    ChangeMovieAction?.Invoke();

                    if (CurrentMovieIndex + 1 >= Dataset.MovieFilePaths.Length)
                    {
                        OnSwitchDataFile?.Invoke(Loss, Error, Accuracy, Dataset.MovieFilePaths[CurrentMovieIndex], Stopwatch);
                        Reset();
                        Gpu.Use = true;
                        break;
                    }

                    CurrentMovieIndex++;

                    var movieIndex = MovieIndex[CurrentMovieIndex].asscalar<int>();
                    var targetFilePath = Dataset.MovieFilePaths[movieIndex];
                    VideoCapture?.Dispose();
                    VideoCapture = new VideoCapture(targetFilePath);

                    if (!VideoCapture.IsOpened())
                    {
                        throw new Exception($"Movie file not found. {targetFilePath}");
                    }

                    _FrameCount = (long)VideoCapture.Get(VideoCaptureProperties.FrameCount);

                    OnSwitchDataFile?.Invoke(Loss, Error, Accuracy, Dataset.MovieFilePaths[CurrentMovieIndex], Stopwatch);
                    continue;
                }

                if (next.Item1 == IterationStatus.Break)
                {
                    _FrameCount = CurrentFrameIndex = Math.Min(CurrentFrameIndex, _FrameCount);
                    ConsoleOut();
                    break;
                }

                Gpu.Use = true;
                yield return (xp.array(x.Select(y => y).ToArray()), xp.array(t.Select(y => y).ToArray()));
                Gpu.Use = false;

                ConsoleOut();

                if (IsRunningFromVisualStudio())
                {
                    Console.SetCursorPosition(0, Console.CursorTop - 1);
                }
                else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                {
                    //一行上の行頭にカーソルを移動
                    Console.Write("\u001b[F");
                }
            }
        }

        private void ConsoleOut()
        {
            if (_FrameCount == 0)
            {
                return;
            }
            Console.OutputEncoding = Encoding.UTF8;
            var strBuilder = new StringBuilder();
            var percentage = (int)((double)CurrentFrameIndex / _FrameCount * 100);
            var percent_len = percentage.ToString().Length;
            strBuilder.Append($"{" ".PadLeft(3 - percent_len)}{percentage.ToString()}%");
            strBuilder.Append($"|");
            for (int _i = 0; _i < 20; _i++)
            {
                if (_i < percentage / 5)
                    strBuilder.Append('█');
                else
                    strBuilder.Append(" ");
            }
            strBuilder.Append("|");
            strBuilder.Append($" {CurrentFrameIndex}/{_FrameCount} {Dataset.MovieFilePaths[MovieIndex[CurrentMovieIndex].asscalar<int>()]}");
            if (Iteration == MaxIter || (CurrentFrameIndex != _FrameCount && IsChildProcess()))
            {
                strBuilder.Append(" ");
            }
            Console.WriteLine(strBuilder.ToString());
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        static bool IsChildProcess()
        {
            int currentProcessId = Process.GetCurrentProcess().Id;
            int parentProcessId = 0;

            try
            {
                using (var currentProcess = Process.GetCurrentProcess())
                {
                    parentProcessId = GetParentProcessId(currentProcess.Handle);
                }
            }
            catch (Exception)
            {
                // 親プロセスのIDが取得できなかった場合は、親プロセスと見なす
                return false;
            }

            return currentProcessId != parentProcessId;
        }

        static int GetParentProcessId(IntPtr processHandle)
        {
            var parentProcessId = 0;
            var processInfo = new PROCESS_BASIC_INFORMATION();

            if (NtQueryInformationProcess(processHandle, 0, ref processInfo,
                    (uint)Marshal.SizeOf(processInfo), out _) == 0)
            {
                parentProcessId = (int)processInfo.InheritedFromUniqueProcessId;
            }

            return parentProcessId;
        }

        static bool IsRunningFromVisualStudio()
        {
            try
            {
                using (var currentProcess = Process.GetCurrentProcess())
                using (var parentProcess = ParentProcessUtilities.GetParentProcess(currentProcess.Id))
                {
                    if (parentProcess != null)
                    {
                        string parentProcessName = Path.GetFileNameWithoutExtension(parentProcess.MainModule.FileName);
                        return parentProcessName.Equals("VsDebugConsole", StringComparison.OrdinalIgnoreCase);
                    }
                }
            }
            catch (Exception)
            {
                // 親プロセスの情報が取得できなかった場合は、Visual Studio以外から実行されていると見なす
                return false;
            }

            return false;
        }

        public void NotifyEvalValues(double loss, double error, double accuracy, Stopwatch sw)
        {
            Loss = loss;
            Error = error;
            Accuracy = accuracy;
            Stopwatch = sw;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct PROCESS_BASIC_INFORMATION
        {
            public IntPtr Reserved1;
            public IntPtr PebBaseAddress;
            public IntPtr Reserved2_0;
            public IntPtr Reserved2_1;
            public IntPtr UniqueProcessId;
            public IntPtr InheritedFromUniqueProcessId;
        }

        [DllImport("ntdll.dll")]
        private static extern int NtQueryInformationProcess(IntPtr processHandle, int processInformationClass,
            ref PROCESS_BASIC_INFORMATION processInformation, uint processInformationLength, out int returnLength);
    }
}
