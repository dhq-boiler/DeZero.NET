namespace DeZero.NET.Log
{
    public class ConsoleLogger : ILogger
    {
        private readonly LogLevel _minimumLevel;
        private readonly bool _isVerbose;

        public ConsoleLogger(LogLevel minimumLevel, bool isVerbose)
        {
            _minimumLevel = minimumLevel;
            _isVerbose = isVerbose;
        }

        public void Log(LogLevel level, string message)
        {
            if (!_isVerbose && level > _minimumLevel) return;

            var timestamp = DateTime.Now.ToString("yyyy-MM-dd(ddd) HH:mm:ss.fff");
            var levelString = level.ToString().ToUpper();
            Console.WriteLine($"{timestamp} [{levelString}] {message}");
        }

        public void LogError(string message)
        {
            Log(LogLevel.Error, message);
        }

        public void LogWarning(string message)
        {
            Log(LogLevel.Warning, message);
        }

        public void LogInfo(string message)
        {
            Log(LogLevel.Info, message);
        }

        public void LogDebug(string message)
        {
            Log(LogLevel.Debug, message);
        }

        public void LogTrace(string message)
        {
            Log(LogLevel.Trace, message);
        }
    }
}
