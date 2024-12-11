using DeZero.NET.Log;

namespace DeZero.NET.Core
{
    public class InstanceTracker<T> where T : class
    {
        private static volatile InstanceTracker<T> _instance;
        private static readonly object _lock = new object();
        private volatile bool _isDisposed;
        private readonly HashSet<WeakReference<T>> _instances = new HashSet<WeakReference<T>>();
        private readonly ILogger _logger;
        public static bool IsEnabled { get; set; } = false;
        public static LogLevel DefaultLogLevel { get; set; } = LogLevel.Info;
        public static bool IsVerbose { get; set; } = false;

        public static InstanceTracker<T> Instance
        {
            get
            {
                if (_instance == null)
                {
                    lock (_lock)
                    {
                        if (_instance == null)
                        {
                            _instance = new InstanceTracker<T>();
                        }
                    }
                }
                return _instance;
            }
        }

        public InstanceTracker()
        {
            _logger = new ConsoleLogger(DefaultLogLevel, isVerbose: IsVerbose);
        }

        public LogLevel LogLevel
        {
            get => _logger.MinimumLevel;
            set => _logger.MinimumLevel = value;
        }
        public static string Filter { get; set; }

        public void Register(T instance)
        {
            lock (_instances)
            {
                // Clean up any dead references first
                _instances.RemoveWhere(weakRef => !weakRef.TryGetTarget(out _));

                // Add the new instance
                _instances.Add(new WeakReference<T>(instance));
            }
        }

        public void Unregister(T instance)
        {
            lock (_instances)
            {
                _instances.RemoveWhere(weakRef =>
                {
                    if (!weakRef.TryGetTarget(out T target))
                        return true;

                    return ReferenceEquals(target, instance);
                });
            }
        }

        public List<T> GetInstances()
        {
            lock (_instances)
            {
                var result = new List<T>();

                // Remove dead references and collect live ones
                _instances.RemoveWhere(weakRef =>
                {
                    if (weakRef.TryGetTarget(out T target))
                    {
                        result.Add(target);
                        return false;
                    }
                    return true;
                });

                return result;
            }
        }

        public void LogMemoryUsage(bool verbose = false, bool ndarray = false)
        {
            if (!IsEnabled) return;

            if (_isDisposed) return;

            verbose |= IsVerbose;

            if ((int)LogLevel >= (int)LogLevel.Debug)
            {
                var dicCount = LogDotNetObjects();

                //コンソールをクリア
                for (int i = 0; i < Console.WindowHeight - 5 - dicCount; i++)
                {
                    Console.WriteLine();
                }
            }
        }

        private int LogDotNetObjects(bool no1_diagonositc = false)
        {
            var instances = GetInstances();
            var ndarrayDic = new Dictionary<string, Dictionary<string, int>>();
            var dic = new Dictionary<string, int>();

            foreach (var obj in instances)
            {
                if (obj is NDarray ndarray && ndarray.Handle != IntPtr.Zero)
                {
                    string shapeStr = default;
                    try
                    {
                        shapeStr = ndarray.shape.ToString();
                    }
                    catch (Exception e)
                    {
                        continue;
                    }

                    if (string.IsNullOrEmpty(shapeStr))
                    {
                        continue;
                    }
#if DEBUG
                    if (ndarrayDic.ContainsKey(shapeStr))
                    {
                        if (ndarrayDic[shapeStr].ContainsKey(ndarray.StackTrace))
                        {
                            ndarrayDic[shapeStr][ndarray.StackTrace]++;
                        }
                        else
                        {
                            ndarrayDic[shapeStr].Add(ndarray.StackTrace, 1);
                        }
                    }
                    else
                    {
                        ndarrayDic.Add(shapeStr, new Dictionary<string, int> { { ndarray.StackTrace, 1 } });
                    }
#endif
                }
            }

            var i = 0;

            foreach (var (shape, stackTraceDic) in ndarrayDic.Where(x => string.IsNullOrEmpty(Filter) || x.Key == Filter))
            {
                foreach (var (stackTrace, count) in stackTraceDic)
                {
                    _logger.LogDebug($"{shape,-50}:\t{count}\t{stackTrace}");
                    i++;
                }
            }

            return i++;
        }
    }
}
