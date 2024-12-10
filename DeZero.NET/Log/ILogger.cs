
namespace DeZero.NET.Log
{
    public interface ILogger
    {
        LogLevel MinimumLevel { get; set; }
        void Log(LogLevel level, string message);
        void LogError(string message);
        void LogWarning(string message);
        void LogInfo(string message);
        void LogDebug(string message);
        void LogTrace(string message);
        void CursorUp(int count = 1);
    }
}
