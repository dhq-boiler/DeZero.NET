using DeZero.NET;
using DeZero.NET.Core;
using DeZero.NET.Extensions;
using DeZero.NET.Functions;
using Python.Runtime;
using System.Runtime.CompilerServices;

namespace MovieFileDataLoaderSampleWorker
{
    public class Conv2dMobileNet : DeZero.NET.Layers.Convolution.Conv2d, IDisposable
    {
        private const int MAX_CACHE_ENTRIES = 100;
        private const int MAX_CACHE_SIZE = 10;
        private const int CLEANUP_INTERVAL_MS = 30000; // 30秒
        private readonly object _cacheLock = new object();
        private readonly Dictionary<string, CacheEntry> _cache = new();
        private readonly TimeSpan _cacheTimeout = TimeSpan.FromMinutes(2);
        private DateTime _lastCleanup = DateTime.UtcNow;
        private int _cacheCount = 0;
        private bool _isDisposed;

        private class CacheEntry
        {
            public NDarray Col { get; set; }
            public Shape Shape { get; set; }
            public DateTime LastAccessed { get; set; }
            public long MemorySize { get; set; }
        }

        public Conv2dMobileNet(int out_channels, int kernel_size, Dtype dtype,
            int stride = 1, int pad = 0, bool nobias = false, int? in_channels = null)
            : base(out_channels, kernel_size, dtype, stride, pad, nobias, in_channels)
        {
            if (W?.Value != null)
            {
                OptimizeWeights();
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private string GetCacheKey(Shape inputShape) =>
            $"{inputShape[0]}_{inputShape[1]}_{inputShape[2]}_{inputShape[3]}";

        private void OptimizeWeights()
        {
            if (W?.Value?.Data?.Value is not null)
            {
                using (Py.GIL()) // Python GILの確保
                {
                    try
                    {
                        using var temp_W = xp.ascontiguousarray(W.Value.Data.Value);
                        using var oldW = W.Value.Data.Value;
                        W.Value.Data.Value = temp_W.copy();
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Weight optimization failed: {ex.Message}");
                    }
                }
            }
        }

        public override Variable[] Forward(params Variable[] xs)
        {
            if (_isDisposed)
                throw new ObjectDisposedException(nameof(Conv2dMobileNet));

            if (!ValidateInputs(xs, out var x))
                return null;

            try
            {
                if (NeedsWeightInitialization(x))
                {
                    InitializeWeights(x);
                }

                PerformCacheCleanupIfNeeded();
                
                using var col = ComputeAndCacheCol(x);

                using var y = ComputeOutput(col);

                return [y.copy()];
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Forward pass failed: {ex.Message}");
                return null;
            }
        }

        private bool TryGetFromCache(string key, Shape shape, out NDarray col)
        {
            col = null;
            var found = _cache.TryGetValue(key, out var entry) &&
                        entry.Shape.Dimensions.SequenceEqual(shape.Dimensions);

            CacheMetrics.RecordAccess(found);

            if (found)
            {
                col = entry.Col.copy();
                entry.LastAccessed = DateTime.UtcNow;
                return true;
            }
            return false;
        }

        private Variable ComputeAndCacheCol(Variable x)
        {
            using var col = Im2col.Invoke(x, (W.Value.Shape[2], W.Value.Shape[3]),
                (Stride.Value, Stride.Value), (Pad.Value, Pad.Value), toMatrix: false);

            return col.copy();
        }

        //private void ManageCache(string key, NDarray col, Shape shape)
        //{
        //    if (_cacheCount >= MAX_CACHE_ENTRIES)
        //    {
        //        RemoveOldestEntries();
        //    }

        //    var memorySize = CalculateArrayMemorySize(col);
        //    var entry = new CacheEntry
        //    {
        //        Col = col.copy(),
        //        Shape = shape,
        //        LastAccessed = DateTime.UtcNow,
        //        MemorySize = memorySize
        //    };

        //    if (_cache.TryGetValue(key, out var oldEntry))
        //    {
        //        oldEntry.Col?.Dispose();
        //        _cache[key] = entry;
        //    }
        //    else
        //    {
        //        _cache.Add(key, entry);
        //        _cacheCount++;
        //    }
        //}

        private void ManageCache(string key, NDarray col, Shape shape)
        {
            // 定期的なキャッシュクリーンアップ
            if (_cacheCount > MAX_CACHE_SIZE)
            {
                CleanupOldestCache();
            }

            // LRUキャッシュの実装
            if (_cache.ContainsKey(key))
            {
                _cache[key].LastAccessed = DateTime.Now;
                return;
            }

            _cache[key] = new CacheEntry
            {
                Col = col.copy(),
                Shape = shape,
                LastAccessed = DateTime.Now
            };
            _cacheCount++;
        }

        private void CleanupOldestCache()
        {
            var oldestEntries = _cache
                .OrderBy(x => x.Value.LastAccessed)
                .Take(_cacheCount / 2); // 半分のキャッシュを削除

            foreach (var entry in oldestEntries)
            {
                entry.Value.Col?.Dispose();
                _cache.Remove(entry.Key);
            }

            _cacheCount = _cache.Count;
            GC.Collect();
        }

        private void RemoveOldestEntries()
        {
            var entriesToRemove = _cache
                .OrderBy(x => x.Value.LastAccessed)
                .Take(_cacheCount / 4) // 25%を削除
                .ToList();

            foreach (var entry in entriesToRemove)
            {
                entry.Value.Col?.Dispose();
                _cache.Remove(entry.Key);
                _cacheCount--;
            }
        }

        private void PerformCacheCleanupIfNeeded()
        {
            var now = DateTime.UtcNow;
            if ((now - _lastCleanup).TotalMilliseconds >= CLEANUP_INTERVAL_MS)
            {
                CleanupCache();
                _lastCleanup = now;
            }
        }

        private void CleanupCache()
        {
            lock (_cacheLock)
            {
                var now = DateTime.UtcNow;
                var keysToRemove = _cache
                    .Where(kvp => now - kvp.Value.LastAccessed > _cacheTimeout)
                    .Select(kvp => kvp.Key)
                    .ToList();

                foreach (var key in keysToRemove)
                {
                    if (_cache.TryGetValue(key, out var entry))
                    {
                        entry.Col?.Dispose();
                        _cache.Remove(key);
                        _cacheCount--;
                    }
                }

                // メモリ使用量が高い場合は追加のクリーンアップを実行
                if (IsMemoryUsageHigh())
                {
                    RemoveOldestEntries();
                }
            }
        }

        private bool IsMemoryUsageHigh()
        {
            using (var memInfo = new GpuMemoryInfo())
            {
                return memInfo.UsedMemoryMB > memInfo.TotalMemoryMB * 0.8; // 80%以上使用している場合
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static long CalculateArrayMemorySize(NDarray array)
        {
            try
            {
                using (Py.GIL())
                {
                    using var array_shape = array.shape;
                    using var array_dtype = array.dtype;
                    long totalElements = array_shape.Dimensions.Select(x => (long)x).Aggregate((a, b) => a * b);
                    int elementSize = array_dtype.ToString().Contains("float32") ? 4 : 8;
                    return totalElements * elementSize;
                }
            }
            catch
            {
                return 0;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool ValidateInputs(Variable[] xs, out Variable x)
        {
            x = null;
            if (xs is null || xs.Length == 0 || xs[0]?.Data?.Value is null)
                return false;

            x = xs[0];
            return true;
        }

        private bool NeedsWeightInitialization(Variable x) =>
            InChannels == null ||
            x.Shape[1] != InChannels.Value ||
            W?.Value?.Data?.Value is null;

        private void InitializeWeights(Variable x)
        {
            InChannels.Value = x.Shape[1];
            W.Value.Data.Value = null;
            _init_W();
            OptimizeWeights();
            WInitialized?.Invoke();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private Variable ComputeOutput(Variable col)
        {
            using (Py.GIL())
            {
                try
                {
                    using var y1 = Tensordot.Invoke(col, W.Value, [1, 2, 3], [1, 2, 3])[0];

                    using var y = Transpose.Invoke(y1, [new Axis([0, 3, 1, 2])])[0];

                    if (b?.Value?.Data?.Value is not null)
                    {
                        using var b_shape = b.Value.Data.Value.shape;
                        using var target_shape = new Shape(1, b_shape[0], 1, 1);
                        using var broadcastedBias = Reshape.Invoke(b.Value, target_shape)[0];
                        using var y_temp = Add.Invoke(y, broadcastedBias).Item1[0];
                        return y_temp.copy();
                    }

                    return y.copy();
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Output computation failed: {ex.Message}");
                    throw;
                }
            }
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_isDisposed)
            {
                if (disposing)
                {
                    lock (_cacheLock)
                    {
                        foreach (var entry in _cache.Values)
                        {
                            entry.Col?.Dispose();
                        }
                        _cache.Clear();
                        _cacheCount = 0;
                    }
                }
                _isDisposed = true;
            }
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        ~Conv2dMobileNet()
        {
            Dispose(false);
        }
    }

    public class CacheMetrics
    {
        private static long _totalAccesses = 0;
        private static long _cacheHits = 0;
        private static readonly object _lockObj = new object();

        public static void RecordAccess(bool isHit)
        {
            lock (_lockObj)
            {
                _totalAccesses++;
                if (isHit) _cacheHits++;

                if (_totalAccesses % 100 == 0) // 100アクセスごとに統計を表示
                {
                    var hitRate = (double)_cacheHits / _totalAccesses * 100;
                    Console.WriteLine($"Cache hit rate: {hitRate:F2}%, Hits: {_cacheHits}, Total: {_totalAccesses}");

                    using (var memInfo = new GpuMemoryInfo())
                    {
                        Console.WriteLine($"GPU Memory Usage: {memInfo.UsedMemoryMB}MB / {memInfo.TotalMemoryMB}MB");
                    }
                }
            }
        }
    }

    public class Conv2dMobileNetFunction : DeZero.NET.Functions.Conv2d
    {
        public Conv2dMobileNetFunction(int stride, int pad) : base(stride, pad)
        {
        }

        public override Variable[] Forward(Params args)
        {
            if (args == null)
                throw new ArgumentNullException(nameof(args));

            var x = args.Get<Variable>("x");
            var W = args.Get<Variable>("W");
            var b = args.Get<Variable>("b");

            if (x?.Data?.Value is null)
                throw new ArgumentException("Input variable x is null or has null data");
            if (W?.Data?.Value is null)
                throw new ArgumentException("Weight variable W is null or has null data");
            if (W.Shape == null || W.Shape.Dimensions.Length < 4)
                throw new ArgumentException("Weight shape is invalid");

            int kernelHeight = W.Shape[2];
            int kernelWidth = W.Shape[3];

            if (kernelHeight <= 0 || kernelWidth <= 0)
                throw new ArgumentException("Invalid kernel dimensions");

            var col = Utils.im2col_array(x, (kernelHeight, kernelWidth), Stride, Pad, to_matrix: false);
            if (col?.Data?.Value is null)
                throw new InvalidOperationException("im2col_array returned null result");

            var y = xp.tensordot(col.Data.Value, W.Data.Value,
                new int[][] { new int[] { 1, 2, 3 }, new int[] { 1, 2, 3 } });

            y = xp.transpose(y, new int[] { 0, 3, 1, 2 });

            if (b?.Data?.Value is not null)
            {
                using var b_shape = b.Data.Value.shape;
                var broadcastedBias = xp.reshape(b.Data.Value, new Shape(1, b_shape[0], 1, 1));
                y = xp.add(y, broadcastedBias);
            }

            return new[] { y.ToVariable(this) };
        }

        public static Variable[] Invoke(Variable x, Variable W, Variable b = null, int stride = 1, int pad = 0)
        {
            return new Conv2dMobileNetFunction(stride, pad).Call(Params.New.SetKeywordArg(x, W, b));
        }
    }
}
