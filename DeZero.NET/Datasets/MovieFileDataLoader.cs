using DeZero.NET.Models;
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
        public NDarray MovieIndex { get; private set; }

        public int CurrentMovieIndex { get; protected set; }

        public long CurrentFrameIndex { get; protected set; }

        public Action ChangeMovieAction { get; set; }

        private VideoCapture VideoCapture { get; set; }

        public long Length => _FrameCount;

        public MovieFileDataLoader(MovieFileDataset dataset, int batchSize, Action changeMovieAction, bool shuffle = true)
        {
            Dataset = dataset;
            BatchSize = batchSize;
            Shuffle = shuffle;
            MaxIter = 1;
            ChangeMovieAction = changeMovieAction;
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

            while (frames.Count < BatchSize)
            {
                if (_buffer.Count == 0)
                {
                    if (!FillBuffer())
                    {
                        // バッファを埋められなかった場合（データセットの終わり）
                        if (frames.Count > 0)
                        {
                            return (IterationStatus.Break, (frames.ToArray(), labels.ToArray()));
                        }
                        return (IterationStatus.Break, (null, null));
                    }
                }

                var (frame, label) = _buffer.Dequeue();
                frames.Add(frame);
                labels.Add(label);
            }

            return (IterationStatus.Continue, (frames.ToArray(), labels.ToArray()));
        }

        private bool FillBuffer()
        {
            if (CurrentFrameIndex == 0)
            {
                int movieIndex = MovieIndex[CurrentMovieIndex].GetData<int>();
                var targetFilePath = Dataset.MovieFilePaths[movieIndex];
                VideoCapture?.Dispose();
                VideoCapture = new VideoCapture(targetFilePath);

                if (!VideoCapture.IsOpened())
                {
                    throw new Exception("Movie file not found.");
                }

                _FrameCount = (long)VideoCapture.Get(VideoCaptureProperties.FrameCount);
            }

            while (_buffer.Count < BatchSize)
            {
                int movieIndex = 0;
                var canContinue = CheckContinue(ref movieIndex);
                if (!canContinue)
                {
                    return false;
                }

                VideoCapture.Set(VideoCaptureProperties.PosFrames, CurrentFrameIndex);
                VideoCapture.Retrieve(out var ndArray);

                movieIndex = MovieIndex[CurrentMovieIndex].GetData<int>();
                var labelNdArray = Dataset.LabelArray[movieIndex][(int)CurrentFrameIndex];

                ndArray = ProcessFrame(ndArray);

                _buffer.Enqueue((ndArray, labelNdArray));

                CurrentFrameIndex++;
            }

            return true;
        }

        public virtual NDarray ProcessFrame(NDarray ndArray)
        {
            return ndArray;
        }

        private bool CheckContinue(ref int movieIndex)
        {
            if (CurrentFrameIndex >= _FrameCount)
            {
                CurrentFrameIndex = 0;
                CurrentMovieIndex++;
                ChangeMovieAction?.Invoke();

                if (CurrentMovieIndex >= Dataset.MovieFilePaths.Length)
                {
                    Reset();
                    if (IsRunningFromVisualStudio())
                    {
                        Console.SetCursorPosition(0, Console.CursorTop + 1);
                    }
                    return false;
                }

                movieIndex = MovieIndex[CurrentMovieIndex].GetData<int>();
                var targetFilePath = Dataset.MovieFilePaths[movieIndex];
                VideoCapture?.Dispose();
                VideoCapture = new VideoCapture(targetFilePath);

                if (!VideoCapture.IsOpened())
                {
                    throw new Exception("Movie file not found.");
                }

                _FrameCount = (long)VideoCapture.Get(VideoCaptureProperties.FrameCount);
            }
            return true;
        }

        public IEnumerator<(NDarray, NDarray)> GetEnumerator()
        {
            if (IsRunningFromVisualStudio())
            {
                Console.CursorVisible = false;
            }
            while (true)
            {
                var next = Next();
                var x = next.Item2.Item1;
                var t = next.Item2.Item2;

                if (x is null && t is null)
                {
                    CurrentFrameIndex = _FrameCount;
                    if (IsRunningFromVisualStudio())
                    {
                        Console.SetCursorPosition(0, Console.CursorTop - 1);
                    }
                    ConsoleOut();
                    break;
                }

                yield return (xp.array(x.Select(y => new NDarray(y.ToCupyNDarray)).ToArray()), xp.array(t.Select(y => new NDarray(y.ToCupyNDarray)).ToArray()));

                ConsoleOut();

                if (IsRunningFromVisualStudio())
                {
                    Console.SetCursorPosition(0, Console.CursorTop - 1);
                }

                if (next.Item1 == IterationStatus.Break)
                {
                    break;
                }
            }
        }

        private void ConsoleOut()
        {
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
            strBuilder.Append($" {CurrentFrameIndex}/{_FrameCount} {Dataset.MovieFilePaths[CurrentMovieIndex]}");
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
