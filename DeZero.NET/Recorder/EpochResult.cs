using DeZero.NET.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeZero.NET.Recorder
{

    public class EpochResult
    {
        public ModelType ModelType { get; set; }
        public int Epoch { get; set; }
        public double TrainLoss { get; set; }
        public double TrainError { get; set; }
        public double TrainAccuracy { get; set; }
        public double TestLoss { get; set; }
        public double TestError { get; set; }
        public double TestAccuracy { get; set; }
        public long ElapsedMilliseconds { get; set; }
        public string TargetDataFile { get; set; }
        public TrainOrTest TrainOrTestType { get; set; }

        public enum TrainOrTest
        {
            Train,
            TrainTotal,
            Test,
            TestTotal
        }
    }
}
