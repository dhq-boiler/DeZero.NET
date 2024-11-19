using DeZero.NET.Core;
using DeZero.NET.Log;
using DeZero.NET.OpenCv;
using DeZero.NET.Processes;
using System.Collections;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

namespace DeZero.NET.Datasets
{
    public class MovieFileDataLoader : IDataProvider
    {
        private readonly ILogger _logger = new ConsoleLogger(LogLevel.Info, false);
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
        public Action<ResultMetrics, string, Stopwatch> OnSwitchDataFile { get; set; }

        private Dictionary<string, long> _fileFrameCounts = new Dictionary<string, long>();

        private ResultMetrics ResultMetrics { get; set; }
        private Stopwatch Stopwatch { get; set; }

        public MovieFileDataLoader(MovieFileDataset dataset, int batchSize, Action changeMovieAction, bool shuffle = true, Action<ResultMetrics, string, Stopwatch> onSwitchDataFile = null)
        {
            Dataset = dataset;
            BatchSize = batchSize;
            Shuffle = shuffle;
            MaxIter = 1;
            ChangeMovieAction = changeMovieAction;
            OnSwitchDataFile = onSwitchDataFile;

            // 初期化時に総フレーム数を計算
            CalculateTotalFrames();
            Reset();
        }

        private void CalculateTotalFrames()
        {
            _fileFrameCounts.Clear();

            for (int i = 0; i < Dataset.MovieFilePaths.Length; i++)
            {
                using (var vc = new VideoCapture(Dataset.MovieFilePaths[i]))
                {
                    if (vc.IsOpened())
                    {
                        var frameCount = (long)vc.Get(VideoCaptureProperties.FrameCount);
                        var labelCount = Dataset.LabelArray[i].len;
                        var effectiveFrames = Math.Min(frameCount, labelCount);
                        _fileFrameCounts[Dataset.MovieFilePaths[i]] = effectiveFrames;
                    }
                }
            }
        }

        protected void Reset()
        {
            Iteration = 0;
            CurrentMovieIndex = 0;
            CurrentFrameIndex = 0;
            _fileFrameCounts.Clear();
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
        private double LocalTime;

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
                if (ConsoleLogger.LastMessage.Contains("%"))
                {
                    if (ProcessUtil.IsChildProcess())
                    {
                        Console.Write(CURSOR_UP);
                    }
                    else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                    {
                        //一行上の行頭にカーソルを移動
                        Console.Write("\u001b[F");
                    }
                    else
                    {
                        Console.SetCursorPosition(0, Console.CursorTop - 1);
                    }
                }
                ConsoleOut();

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
            // MovieIndexの範囲チェック
            if (CurrentMovieIndex >= MovieIndex.len || CurrentMovieIndex < 0)
            {
                return IterationStatus.Break;
            }

            // refパラメータを使用
            movieIndex = MovieIndex[CurrentMovieIndex].asscalar<int>();
            var currentFilePath = Dataset.MovieFilePaths[movieIndex];

            // ファイルパスのキーチェック
            if (!_fileFrameCounts.ContainsKey(currentFilePath))
            {
                // キーが存在しない場合は、その場でフレーム数を計算して追加
                using (var vc = new VideoCapture(currentFilePath))
                {
                    if (vc.IsOpened())
                    {
                        var frameCount = (long)vc.Get(VideoCaptureProperties.FrameCount);
                        var labelCount = Dataset.LabelArray[movieIndex].len;
                        var effectiveFrames = Math.Min(frameCount, labelCount);
                        _fileFrameCounts[currentFilePath] = effectiveFrames;
                    }
                    else
                    {
                        // ファイルが開けない場合はエラー
                        throw new Exception($"Movie file could not be opened: {currentFilePath}");
                    }
                }
            }

            var currentFileFrames = _fileFrameCounts[currentFilePath];

            if (CurrentFrameIndex >= currentFileFrames)
            {
                if (CurrentMovieIndex >= MovieIndex.len - 1)
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

        private long CalculateEffectiveFrames(string filePath, int movieIndex)
        {
            using (var vc = new VideoCapture(filePath))
            {
                if (!vc.IsOpened())
                {
                    throw new Exception($"Movie file could not be opened: {filePath}");
                }

                var frameCount = (long)vc.Get(VideoCaptureProperties.FrameCount);
                var labelCount = Dataset.LabelArray[movieIndex].len;
                return Math.Min(frameCount, labelCount);
            }
        }

        private void EnsureFrameCount(string filePath, int movieIndex)
        {
            if (!_fileFrameCounts.ContainsKey(filePath))
            {
                var effectiveFrames = CalculateEffectiveFrames(filePath, movieIndex);
                _fileFrameCounts[filePath] = effectiveFrames;
            }
        }

        public IEnumerator<(NDarray, NDarray)> GetEnumerator()
        {
            if (ProcessUtil.IsRunningFromVisualStudio())
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
                    Gpu.Use = true;
                    yield return (xp.array(x.Select(y => y).ToArray()), xp.array(t.Select(y => y).ToArray()));
                    Gpu.Use = false;

                    ConsoleOut();
                    CurrentFrameIndex = 0;
                    ChangeMovieAction?.Invoke();

                    if (CurrentMovieIndex + 1 >= Dataset.MovieFilePaths.Length)
                    {
                        OnSwitchDataFile?.Invoke(ResultMetrics, Dataset.MovieFilePaths[CurrentMovieIndex], Stopwatch);
                        // 必要な変数のみリセット
                        CurrentMovieIndex = 0;
                        CurrentFrameIndex = 0;
                        MovieIndex?.Dispose();
                        if (Shuffle)
                        {
                            MovieIndex = xp.random.permutation(Dataset.MovieFilePaths.Length);
                        }
                        else
                        {
                            MovieIndex = xp.arange(Dataset.MovieFilePaths.Length);
                        }
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

                    OnSwitchDataFile?.Invoke(ResultMetrics, Dataset.MovieFilePaths[CurrentMovieIndex], Stopwatch);
                    continue;
                }

                if (next.Item1 == IterationStatus.Break)
                {
                    ConsoleOut();
                    break;
                }

                Gpu.Use = true;
                yield return (xp.array(x.Select(y => y).ToArray()), xp.array(t.Select(y => y).ToArray()));
                Gpu.Use = false;

                if (ConsoleLogger.LastMessage.Contains("%"))
                {
                    if (ProcessUtil.IsRunningFromVisualStudio())
                    {
                        Console.SetCursorPosition(0, Console.CursorTop - 1);
                    }
                    else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                    {
                        //一行上の行頭にカーソルを移動
                        Console.Write("\u001b[F");
                    }
                }

                ConsoleOut();
            }
        }


        private const string CURSOR_UP = "__CURSOR_UP__";
        private const string PROGRESS_START = "__PROGRESS_START__";
        private const string PROGRESS_END = "__PROGRESS_END__";

        protected void WriteProgress(string message, bool isStart = false, bool isEnd = false)
        {
            //if (isStart)
            //{
            //    if (ProcessUtil.IsChildProcess())
            //    {
            //        _logger.LogInfo($"{message}");
            //    }
            //    else
            //    {
            //        _logger.LogInfo($"{message}{PROGRESS_START}");
            //    }
            //}
            //else if (isEnd)
            //{
            //    if (ProcessUtil.IsChildProcess())
            //    {
            //        _logger.LogInfo($"{message}");
            //    }
            //    else
            //    {
            //        _logger.LogInfo($"{message}{PROGRESS_END}");
            //    }
            //}
            //else
            {
                _logger.LogInfo($"{message}");  // スペースを追加して上書き行を示す
                //if (ProcessUtil.IsChildProcess())
                //{
                //    _logger.LogInfo(CURSOR_UP);  // 親プロセスに上カーソル移動を指示
                //}
            }
        }

        private void ConsoleOut()
        {
            try
            {
                // MovieIndexの範囲チェック
                if (CurrentMovieIndex >= MovieIndex.len || CurrentMovieIndex < 0)
                {
                    return;
                }

                var currentMovieIndex = MovieIndex[CurrentMovieIndex].asscalar<int>();
                var currentFilePath = Dataset.MovieFilePaths[currentMovieIndex];

                // ファイルパスのキーチェック
                if (!_fileFrameCounts.ContainsKey(currentFilePath))
                {
                    EnsureFrameCount(currentFilePath, currentMovieIndex);
                }

                var currentFileFrames = _fileFrameCounts[currentFilePath];

                // 現在のファイルに対する進捗率を計算
                var percentage = (int)((double)CurrentFrameIndex / currentFileFrames * 100);

                Console.OutputEncoding = Encoding.UTF8;
                var strBuilder = new StringBuilder();

                // パーセンテージ表示の整形
                var percent_len = percentage.ToString().Length;
                strBuilder.Append($"{" ".PadLeft(3 - percent_len)}{percentage}%");

                // プログレスバーの表示
                strBuilder.Append("|");
                for (int i = 0; i < 20; i++)
                {
                    strBuilder.Append(i < percentage / 5 ? "█" : " ");
                }
                strBuilder.Append("|");

                // 進捗の詳細表示（現在のファイルの進捗）
                strBuilder.Append($" {CurrentFrameIndex}/{currentFileFrames} ");
                strBuilder.Append(currentFilePath);
                strBuilder.Append($" ({LocalTime:N1}s)");

                WriteProgress(strBuilder.ToString(), CurrentFrameIndex == 0, CurrentFrameIndex == currentFileFrames);
                //_logger.LogInfo(strBuilder.ToString());
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error in ConsoleOut: {ex.Message}");
                // エラーをスローせず、表示をスキップ
            }
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public void SetResultMetricsAndStopwatch(ResultMetrics resultMetrics, Stopwatch sw)
        {
            ResultMetrics = resultMetrics;
            Stopwatch = sw;
        }

        public void SetLocalStopwatch(Stopwatch sw)
        {
            LocalTime = (sw.ElapsedMilliseconds / 1000d);
        }
    }
}
