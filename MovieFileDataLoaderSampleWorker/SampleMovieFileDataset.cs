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

        public override string[] MovieFilePaths => [
            @"Z:\Videos\2024-10-28\2024-10-28 22-55-40.mp4",
            //@"Z:\Videos\2024-10-28\2024-10-28 22-55-40 - コピー.mp4",
            //@"Z:\Videos\2024-10-28\2024-10-28 22-55-40 - コピー (2).mp4",
            //@"Z:\Videos\2024-10-28\2024-10-28 22-55-40 - コピー (3).mp4",
            //@"Z:\Videos\2024-10-28\2024-10-28 22-55-40 - コピー (4).mp4",
            //@"Z:\Videos\2024-10-28\2024-10-28 22-55-40 - コピー (5).mp4",
        ];

        public override string[] LabelFilePaths => [
            @"Z:\Videos\2024-10-28\label_data_2024-10-28_22-55-40.npz",
            //@"Z:\Videos\2024-10-28\label_data_2024-10-28_22-55-40 - コピー.npz",
            //@"Z:\Videos\2024-10-28\label_data_2024-10-28_22-55-40 - コピー (2).npz",
            //@"Z:\Videos\2024-10-28\label_data_2024-10-28_22-55-40 - コピー (3).npz",
            //@"Z:\Videos\2024-10-28\label_data_2024-10-28_22-55-40 - コピー (4).npz",
            //@"Z:\Videos\2024-10-28\label_data_2024-10-28_22-55-40 - コピー (5).npz",
        ];

        public override string[] LabelFileNpzIndex => [
            "label_data",
            //"label_data",
            //"label_data",
            //"label_data",
            //"label_data",
            //"label_data",
        ];
    }
}
