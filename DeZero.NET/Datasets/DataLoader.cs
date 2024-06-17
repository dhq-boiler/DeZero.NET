using System.Collections;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

namespace DeZero.NET.Datasets
{
    public class DataLoader : IEnumerable<(NDarray, NDarray)>
    {
        public Dataset Dataset { get; }
        public int BatchSize { get; }
        public bool Shuffle { get; }
        public int DataSize { get; }
        public double MaxIter { get; }
        public int Iteration { get; protected set; }
        public NDarray Index { get; private set; }

        public DataLoader(Dataset dataset, int batch_size, bool shuffle = true)
        {
            Dataset = dataset;
            BatchSize = batch_size;
            Shuffle = shuffle;
            DataSize = dataset.Length;
            MaxIter = Math.Ceiling((double)DataSize / batch_size);
            Reset();
        }

        protected void Reset()
        {
            Iteration = 0;
            Index?.Dispose();
            if (Shuffle)
            {
                Index = xp.random.permutation(Dataset.Length);
            }
            else
            {
                Index = xp.arange(Dataset.Length);
            }
        }

        public virtual (IterationStatus, (NDarray, NDarray)) Next()
        {
            if (Iteration >= MaxIter)
            {
                Reset();
                return (IterationStatus.Break, (null, null));
            }

            var (i, batch_size) = (Iteration, BatchSize);
            var batch_index = Index[new Slice(i * batch_size, (i + 1) * batch_size)];
            using var z = batch_index.flatten();
            var c = z.GetData<int[]>();
            var batch = c.Select(i => Dataset[i]).ToArray();

            var x = xp.array(batch.Select(example =>
            {
                using var reshape = example.Item1.reshape(1, 28, 28);
                return reshape.copy();
            }).ToArray());
            var t = xp.array(batch.Select(example => example.Item2.copy()).ToArray());

            Iteration += 1;

            //カーソルを非表示にする
            if (!IsChildProcess())
            {
                Console.CursorVisible = false;

                if (Iteration > 1)
                {
                    Console.SetCursorPosition(0, Console.CursorTop - 1);
                }
            }

            Console.OutputEncoding = Encoding.UTF8;
            var strBuilder = new StringBuilder();
            var percentage = (int)(Iteration / MaxIter * 100);
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

            if (!IsChildProcess())
            {
                if (Iteration == MaxIter)
                {
                    Console.SetCursorPosition(0, Console.CursorTop - 1);
                }
            }

            return (IterationStatus.Continue, (x, t));
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
