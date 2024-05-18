﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DeZero.NET.Exceptions;

namespace DeZero.NET.Datasets
{
    public class SeqDataLoader : DataLoader
    {
        public SeqDataLoader(Dataset dataset, int batch_size)
            : base(dataset, batch_size, false)
        {
        }

        public override (NDarray, NDarray) Next()
        {
            if (Iteration >= MaxIter)
            {
                Reset();
                throw new StopIterationException();
            }

            var jump = (int)(DataSize / BatchSize);
            //var (i, batch_size) = (Iteration, BatchSize);
            var batch_index = Enumerable.Range(0, BatchSize).Select(i => (i * jump + Iteration) % DataSize).ToArray();
            var batch = batch_index.Select(i => Dataset[i]).ToList();

            var x = xp.array(batch.Select(example => example.Item1).ToArray());
            var t = xp.array(batch.Select(example => example.Item2).ToArray());

            Iteration += 1;
            return (x, t);
        }
    }
}
