using System.Collections;

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

            var x = xp.array(batch.Select(example => example.Item1.reshape(1, 28, 28)).ToArray());
            var t = xp.array(batch.Select(example => example.Item2).ToArray());

            Iteration += 1;
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
    }
}
