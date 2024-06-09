using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeZero.NET.Recorder
{

    public class EpochResult
    {
        public int Epoch { get; set; }
        public double TrainLoss { get; set; }
        public double TrainAccuracy { get; set; }
        public double TestLoss { get; set; }
        public double TestAccuracy { get; set; }
        public long ElapsedMilliseconds { get; set; }
    }
}
