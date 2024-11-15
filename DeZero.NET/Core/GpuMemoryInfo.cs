using Python.Runtime;

namespace DeZero.NET.Core
{
    /// <summary>
    /// メモリ情報を取得するためのヘルパークラス
    /// </summary>
    public class GpuMemoryInfo : IDisposable
    {
        public long TotalMemoryMB { get; }
        public long UsedMemoryMB { get; }

        public GpuMemoryInfo()
        {
            using (Py.GIL())
            {
                dynamic cupy = Py.Import("cupy");
                dynamic mempool = cupy.get_default_memory_pool();
                TotalMemoryMB = Parse(mempool.total_bytes()) / (1024 * 1024);
                UsedMemoryMB = Parse(mempool.used_bytes()) / (1024 * 1024);
            }
        }

        private long Parse(PyObject pyObject)
        {
            return long.Parse(pyObject.ToString());
        }

        public void Dispose()
        {
            // 必要に応じてリソースを解放
        }
    }
}
