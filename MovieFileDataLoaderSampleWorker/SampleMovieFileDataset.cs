using DeZero.NET.Datasets;
using DeZero.NET.Transforms;

namespace MovieFileDataLoaderSampleWorker
{
    internal class SampleMovieFileDataset : MovieFileDataset
    {
        public SampleMovieFileDataset(bool train = true, Transform transform = null, Transform targetTransform = null)
            : base(train, transform, targetTransform)
        {
        }

        public override string[] MovieFilePaths => [@"C:\Users\boiler\Downloads\2024-07-24_22-21-56_01_ドスジャグラス1320.50.mp4"];

        public override string[] LabelFilePaths => [@"C:\Users\boiler\Downloads\label_data_2024-07-24_22-21-56_01_ドスジャグラス1320.50.npz"];

        public override string[] LabelFileNpzIndex => ["arr_0"];
    }
}
