using OpenCvSharp;
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
        public NDarray MovieIndex { get; private set; }

        public int CurrentMovieIndex { get; protected set; }

        public long CurrentFrameIndex { get; protected set; }

        public MovieFileDataLoader(MovieFileDataset dataset, bool shuffle = true)
        {
            Dataset = dataset;
            Shuffle = shuffle;
            MaxIter = 1;
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

        public virtual (IterationStatus, (NDarray, NDarray)) Next()
        {
            if (CurrentFrameIndex >= _FrameCount)
            {
                CurrentFrameIndex = 0;
                CurrentMovieIndex++;
                if (CurrentMovieIndex >= Dataset.MovieFilePaths.Length)
                {
                    Reset();
                    return (IterationStatus.Break, (null, null));
                }
            }

            var movieIndex = MovieIndex[CurrentMovieIndex].GetData<int>();
            var targetFilePath = Dataset.MovieFilePaths[movieIndex];
            using var vc = new VideoCapture(targetFilePath);
            if (!vc.IsOpened())
            {
                throw new Exception("Movie file not found.");
            }

            // フレーム数を取得
            _FrameCount = (long)vc.Get(VideoCaptureProperties.FrameCount);

            // 任意のフレームに移動
            vc.Set(VideoCaptureProperties.PosFrames, CurrentFrameIndex);

            using var mat = vc.RetrieveMat();

            //チャンネルを連結

            // 1次元配列として取得
            Vec3b[] array = new Vec3b[mat.Total() * mat.Channels()];
            mat.GetArray(out array);

            var array3ch = Vec3bArrayToByteArray(array);

            // 1次元配列を3次元配列に変換
            byte[,,] array3D = new byte[mat.Rows, mat.Cols, mat.Channels()];
            Buffer.BlockCopy(array3ch, 0, array3D, 0, array.Length);

            // NDarrayに変換
            NDarray ndArray = xp.array(array3D);

            var labelNdArray = Dataset.LabelArray[movieIndex][(int)CurrentFrameIndex];

            CurrentFrameIndex++;

            Console.OutputEncoding = Encoding.UTF8;
            var strBuilder = new StringBuilder();
            var percentage = (int)(CurrentFrameIndex / _FrameCount * 100);
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
            strBuilder.Append($" {Iteration}/{MaxIter}");
            if (Iteration == MaxIter || IsChildProcess())
            {
                strBuilder.Append(" ");
            }
            Console.WriteLine(strBuilder.ToString());

            return (IterationStatus.Continue, (ndArray, labelNdArray));
        }

        public static byte[] Vec3bArrayToByteArray(Vec3b[] vec3bArray)
        {
            // 結果のbyte配列のサイズを計算（Vec3bの要素数 * 3）
            byte[] result = new byte[vec3bArray.Length * 3];

            for (int i = 0; i < vec3bArray.Length; i++)
            {
                // Vec3bの各要素をbyte配列に順番に格納
                result[i * 3] = vec3bArray[i].Item0;
                result[i * 3 + 1] = vec3bArray[i].Item1;
                result[i * 3 + 2] = vec3bArray[i].Item2;
            }

            return result;
        }

        public IEnumerator<(NDarray, NDarray)> GetEnumerator()
        {
            while (true)
            {
                var next = Next();
                if (next.Item1 == IterationStatus.Break)
                {
                    break;
                }
                yield return next.Item2;
            }
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
