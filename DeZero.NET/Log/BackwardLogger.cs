using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeZero.NET.Log
{
    public class BackwardLogger : IDisposable
    {
        private readonly StreamWriter _writer;
        private static BackwardLogger _instance;
        private readonly object _lockObject = new object();

        private BackwardLogger()
        {
            var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            var logDir = "backward_logs";
            Directory.CreateDirectory(logDir);
            _writer = new StreamWriter(Path.Combine(logDir, $"backward_{timestamp}.log"), true);
        }

        public static BackwardLogger Instance
        {
            get
            {
                if (_instance == null)
                {
                    _instance = new BackwardLogger();
                }
                return _instance;
            }
        }

        public void Log(string message)
        {
            lock (_lockObject)
            {
                _writer.WriteLine($"{DateTime.Now:HH:mm:ss.fff} {message}");
                _writer.Flush(); // 即座に書き込む
            }
        }

        public void Dispose()
        {
            _writer?.Dispose();
        }
    }
}
