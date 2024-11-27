using DeZero.NET.Extensions;
using System.Collections.Concurrent;

namespace DeZero.NET.Core;
public class BatchProcessingScope : IDisposable
{
    private readonly int _batchSize;
    private readonly ConcurrentBag<Variable> _intermediateResults;
    private readonly object _syncLock = new object();
    private bool _isDisposed;

    public BatchProcessingScope(int batchSize)
    {
        _batchSize = batchSize;
        _intermediateResults = new ConcurrentBag<Variable>();
    }

    public Variable ProcessBatch(Variable input, Func<Variable, Variable> processor)
    {
        if (input.Shape[0] <= _batchSize)
        {
            return processor(input);
        }

        var batches = SplitIntoBatches(input);
        var results = new ConcurrentBag<Variable>();

        Parallel.ForEach(batches, batch =>
        {
            var result = processor(batch);
            results.Add(result);
            TrackIntermediate(batch);
        });

        return ConcatenateBatches(results);
    }

    private IEnumerable<Variable> SplitIntoBatches(Variable input)
    {
        var batchCount = (int)Math.Ceiling(input.Shape[0] / (double)_batchSize);
        for (int i = 0; i < batchCount; i++)
        {
            var start = i * _batchSize;
            var end = Math.Min((i + 1) * _batchSize, input.Shape[0]);
            yield return ExtractBatch(input, start, end);
        }
    }

    private Variable ExtractBatch(Variable input, int start, int end)
    {
        using var scope = new ComputationScope();
        var slice = input.Data.Value.Slice(new[] { new Slice(start, end) });
        return slice.ToVariable();
    }

    private Variable ConcatenateBatches(ConcurrentBag<Variable> results)
    {
        var arrays = results.Select(r => r.Data.Value).ToArray();
        return xp.concatenate(arrays, axis: 0).ToVariable();
    }

    public void TrackIntermediate(Variable variable)
    {
        if (!_isDisposed)
        {
            _intermediateResults.Add(variable);
        }
    }

    public void Dispose()
    {
        if (_isDisposed) return;

        lock (_syncLock)
        {
            if (_isDisposed) return;
            foreach (var result in _intermediateResults)
            {
                result?.Dispose();
            }
            _intermediateResults.Clear();
            _isDisposed = true;
        }
        GC.SuppressFinalize(this);
    }
}