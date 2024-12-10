using DeZero.NET.Processes;
using System.Runtime.InteropServices;

namespace DeZero.NET.Log
{
    public class ConsoleLogger : ILogger
    {
        private const string CURSOR_UP = "__CURSOR_UP__";
        private const string PROGRESS_START = "__PROGRESS_START__";
        private const string PROGRESS_END = "__PROGRESS_END__";

        public LogLevel MinimumLevel { get; set; }
        private readonly bool _isVerbose;
        private int? _progressStartRow = null;
        private bool _isInProgress = false;
        private int? _progressMessageLength = null;

        // 現在のメッセージとその種類を保持
        private record MessageInfo(string Message, MessageType Type);
        private static MessageInfo? _lastMessage = null;

        public static string LastMessage => _lastMessage?.Message;

        private enum MessageType
        {
            Normal,
            ProgressStart,
            ProgressComplete,
            ProgressFailed
        }

        public ConsoleLogger(LogLevel minimumLevel, bool isVerbose)
        {
            MinimumLevel = minimumLevel;
            _isVerbose = isVerbose;
        }

        public void Log(LogLevel level, string message)
        {
            if (!_isVerbose && level > MinimumLevel) return;

            var timestamp = DateTime.Now.ToString("yyyy-MM-dd(ddd) HH:mm:ss.fff");
            var levelString = level.ToString().ToUpper();
            var newMessage = $"{timestamp} [{levelString}] {message}";

            // 前の行が進捗中のメッセージだった場合は改行を入れる
            if (_lastMessage?.Type == MessageType.ProgressStart)
            {
                Console.WriteLine();
            }

            Console.WriteLine(newMessage);
            _lastMessage = new MessageInfo(newMessage, MessageType.Normal);
        }

        public IProgressScope BeginProgress(string message)
        {
            if (_isInProgress)
            {
                throw new InvalidOperationException("Nested progress logging is not supported");
            }

            // 前の行が進捗中のメッセージだった場合は改行を入れる
            if (_lastMessage?.Type == MessageType.ProgressStart)
            {
                Console.WriteLine();
            }

            var timestamp = DateTime.Now.ToString("yyyy-MM-dd(ddd) HH:mm:ss.fff");
            var newMessage = $"{timestamp} [INFO] {message}";
            if (ProcessUtil.IsChildProcess() && !ProcessUtil.IsRunningFromVisualStudio())
            {
                Console.Write($"{PROGRESS_START}{timestamp} [INFO] {message}");
            }
            else
            {
                Console.Write($"{timestamp} [INFO] {message}");
            }
            _lastMessage = new MessageInfo(newMessage, MessageType.ProgressStart);
            //_progressStartRow = Console.CursorTop;
            //_progressMessageLength = newMessage.Length;
            _isInProgress = true;

            return new ProgressScope(this);
        }

        private void CompleteProgress(string message = "Completed.", bool isError = false)
        {
            if (!_isInProgress/* || !_progressStartRow.HasValue || !_progressMessageLength.HasValue*/) return;


            // 進捗開始位置に戻り、完了メッセージを表示
            var prefix = isError ? "Failed: " : "";
            if (ProcessUtil.IsChildProcess() && !ProcessUtil.IsRunningFromVisualStudio())
            {
                Console.WriteLine($"{PROGRESS_END} {prefix}{message}");
            }
            else
            {
                Console.Write($"{prefix}{message}");
            }
            _isInProgress = false;

            //var currentRow = Console.CursorTop;

            //// 前のメッセージが進捗開始メッセージでない場合は改行を入れる
            //bool needsNewLine = _lastMessage?.Type != MessageType.ProgressStart;
            //if (needsNewLine || (currentRow == _progressStartRow && isError))
            //{
            //    Console.WriteLine();
            //    currentRow = Console.CursorTop;
            //}

            //// 進捗開始位置に戻る
            //Console.SetCursorPosition(_progressMessageLength.Value, _progressStartRow.Value);

            //// 完了またはエラーメッセージを書き込む
            //var prefix = isError ? "Failed: " : string.Empty;
            //var completionMessage = $"{prefix}{message}";
            //Console.WriteLine(completionMessage);
            //_lastMessage = new MessageInfo(
            //    _lastMessage?.Message + completionMessage,
            //    isError ? MessageType.ProgressFailed : MessageType.ProgressComplete
            //);

            //// 元の位置に戻る（エラーメッセージの後の位置）
            //Console.SetCursorPosition(0, needsNewLine ? currentRow : _progressStartRow.Value + 1);

            //_progressStartRow = null;
            //_isInProgress = false;
        }

        public void LogError(string message) => Log(LogLevel.Error, message);
        public void LogWarning(string message) => Log(LogLevel.Warning, message);
        public void LogInfo(string message) => Log(LogLevel.Info, message);
        public void LogDebug(string message) => Log(LogLevel.Debug, message);
        public void LogTrace(string message) => Log(LogLevel.Trace, message);

        public void CursorUp(int count = 1)
        {
            if (ProcessUtil.IsChildProcess() && !ProcessUtil.IsRunningFromVisualStudio())
            {
                for (int i = 0; i < count; i++)
                {
                    Console.Write($"{CURSOR_UP}");
                }
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                Console.CursorTop -= count;
            }
        }

        private class ProgressScope : IProgressScope
        {
            private readonly ConsoleLogger _logger;
            private bool _isCompleted = false;

            public ProgressScope(ConsoleLogger logger)
            {
                _logger = logger;
            }

            public void Complete(string message = "Completed")
            {
                if (_isCompleted) return;
                _logger.CompleteProgress(message);
                _isCompleted = true;
            }

            public void Failed(string message)
            {
                if (_isCompleted) return;
                _logger.CompleteProgress(message, isError: true);
                _isCompleted = true;
            }

            public void Dispose()
            {
                if (!_isCompleted)
                {
                    Complete();
                }
            }
        }
    }

    public static class ConsoleLoggerExtensions
    {
        public static IProgressScope BeginProgress(this ILogger logger, string message)
        {
            if (logger is ConsoleLogger consoleLogger)
            {
                return consoleLogger.BeginProgress(message);
            }
            logger.LogInfo(message);
            return new DummyScope();
        }

        private class DummyScope : IProgressScope
        {
            public void Complete(string message = "Completed.")
            {
                throw new NotImplementedException();
            }

            public void Dispose() { }

            public void Failed(string message)
            {
                throw new NotImplementedException();
            }
        }
    }

    public interface IProgressScope : IDisposable
    {
        void Complete(string message = "Completed.");
        void Failed(string message);
    }
}