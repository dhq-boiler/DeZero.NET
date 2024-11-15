using DeZero.NET.Datasets;
using DeZero.NET.Transforms;

namespace MHWGoldCrownModelTrainWorker
{
    public class GoldCrownTestDataset : MovieFileDataset
    {
        public GoldCrownTestDataset(Transform transform = null, Transform target_transform = null)
            : base(false, transform, target_transform)
        {
        }

        public override string[] MovieFilePaths => [
            Path.Combine("Datasets", "Movies", "Test", "2024-07-24_22-21-56_01_ドスジャグラス1320.50_1.mp4"),
            Path.Combine("Datasets", "Movies", "Test", "2024-10-27_14-31-09_1.mp4"),
            Path.Combine("Datasets", "Movies", "Test", "2024-10-27_14-38-26_1.mp4"),
            Path.Combine("Datasets", "Movies", "Test", "2024-10-27_14-47-28_1.mp4"),
            Path.Combine("Datasets", "Movies", "Test", "2024-10-27_14-53-10_1.mp4"),
        ];

        public override string[] LabelFilePaths => [
            Path.Combine("Datasets", "Labels", "Test", "label_data_2024-07-24_22-21-56_01_ドスジャグラス1320.50.npz"),
            Path.Combine("Datasets", "Labels", "Test", "label_data_2024-10-27_14-31-09.npz"),
            Path.Combine("Datasets", "Labels", "Test", "label_data_2024-10-27_14-38-26.npz"),
            Path.Combine("Datasets", "Labels", "Test", "label_data_2024-10-27_14-47-28.npz"),
            Path.Combine("Datasets", "Labels", "Test", "label_data_2024-10-27_14-53-10.npz"),
        ];

        public override string[] LabelFileNpzIndex => [
            "arr_0",
            "label_data",
            "label_data",
            "label_data",
            "label_data",
        ];
    }
}
