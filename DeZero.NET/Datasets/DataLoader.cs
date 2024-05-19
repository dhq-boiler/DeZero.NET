﻿using DeZero.NET.Exceptions;
using System.Collections;
using System.Data;

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
            if (Shuffle)
            {
                Index = xp.random.permutation(Dataset.Length);
            }
            else
            {
                Index = xp.arange(Dataset.Length);
            }
        }

        public virtual (NDarray, NDarray) Next()
        {
            if (Iteration >= MaxIter)
            {
                Reset();
                throw new StopIterationException();
            }

            var (i, batch_size) = (Iteration, BatchSize);
            var batch_index = Index[new Slice(i * batch_size, (i + 1) * batch_size)];
            var z = batch_index.flatten();
            var c = z.GetData<int[]>();
            var batch = c.Select(i => Dataset[i]).ToArray();

            var x = xp.array(batch.Select(example => example.Item1).ToArray());
            var t = xp.array(batch.Select(example => example.Item2).ToArray());

            Iteration += 1;
            return (x, t);
        }

        public IEnumerator<(NDarray, NDarray)> GetEnumerator()
        {
            List<(NDarray, NDarray)> ret = new();

            while (true)
            {
                try
                {
                    ret.Add(Next());
                }
                catch (StopIterationException)
                {
                    return ret.GetEnumerator();
                }
            }
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }
}