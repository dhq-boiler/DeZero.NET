using DeZero.NET.Datasets;
using DeZero.NET.Transforms;

namespace MHWGoldCrownModelTrainWorker
{
    public class GoldCrownTrainDataset : MovieFileDataset
    {
        public GoldCrownTrainDataset(Transform transform = null, Transform target_transform = null)
            : base(true, transform, target_transform)
        {
        }

        public override string[] MovieFilePaths => [
            Path.Combine("Datasets", "Movies", "Train", "2024-10-20_13-48-33_1.mp4"),
            //Path.Combine("Datasets", "Movies", "Train", "2024-10-20_14-00-37_1.mp4"),
            //Path.Combine("Datasets", "Movies", "Train", "2024-10-20_14-11-15_1.mp4"),
            //Path.Combine("Datasets", "Movies", "Train", "2024-10-20_14-22-48_1.mp4"),
            //Path.Combine("Datasets", "Movies", "Train", "2024-10-20_14-42-33_1.mp4"),
            //Path.Combine("Datasets", "Movies", "Train", "2024-10-20_15-08-35_1.mp4"),
            //Path.Combine("Datasets", "Movies", "Train", "2024-10-20_15-36-01_1.mp4"),
            //Path.Combine("Datasets", "Movies", "Train", "2024-10-20_16-05-12_1.mp4"),
            //Path.Combine("Datasets", "Movies", "Train", "2024-10-20_17-19-03_1.mp4"),
            //Path.Combine("Datasets", "Movies", "Train", "2024-10-20_17-29-15_1.mp4"),
            //Path.Combine("Datasets", "Movies", "Train", "2024-10-20_17-58-15_1.mp4"),
            //Path.Combine("Datasets", "Movies", "Train", "2024-10-20_19-28-47_1.mp4"),
            //Path.Combine("Datasets", "Movies", "Train", "2024-10-20_20-17-45_1.mp4"),
            //Path.Combine("Datasets", "Movies", "Train", "2024-10-20_21-41-33_1.mp4"),
            //Path.Combine("Datasets", "Movies", "Train", "2024-10-20_21-57-10_1.mp4"),
            //Path.Combine("Datasets", "Movies", "Train", "2024-10-20_22-53-34_1.mp4"),
            //Path.Combine("Datasets", "Movies", "Train", "2024-10-21_20-46-45_1.mp4"),
            //Path.Combine("Datasets", "Movies", "Train", "2024-10-21_21-05-06_1.mp4"),
            //Path.Combine("Datasets", "Movies", "Train", "2024-10-21_21-39-54_1.mp4"),
        ];

        public override string[] LabelFilePaths => [
            Path.Combine("Datasets", "Labels", "Train", "label_data_2024-10-20_13-48-33.npz"),
            //Path.Combine("Datasets", "Labels", "Train", "label_data_2024-10-20_14-00-37.npz"),
            //Path.Combine("Datasets", "Labels", "Train", "label_data_2024-10-20_14-11-15.npz"),
            //Path.Combine("Datasets", "Labels", "Train", "label_data_2024-10-20_14-22-48.npz"),
            //Path.Combine("Datasets", "Labels", "Train", "label_data_2024-10-20_14-42-33.npz"),
            //Path.Combine("Datasets", "Labels", "Train", "label_data_2024-10-20_15-08-35.npz"),
            //Path.Combine("Datasets", "Labels", "Train", "label_data_2024-10-20_15-36-01.npz"),
            //Path.Combine("Datasets", "Labels", "Train", "label_data_2024-10-20_16-05-12.npz"),
            //Path.Combine("Datasets", "Labels", "Train", "label_data_2024-10-20_17-19-03.npz"),
            //Path.Combine("Datasets", "Labels", "Train", "label_data_2024-10-20_17-29-15.npz"),
            //Path.Combine("Datasets", "Labels", "Train", "label_data_2024-10-20_17-58-15.npz"),
            //Path.Combine("Datasets", "Labels", "Train", "label_data_2024-10-20_19-28-47.npz"),
            //Path.Combine("Datasets", "Labels", "Train", "label_data_2024-10-20_20-17-45.npz"),
            //Path.Combine("Datasets", "Labels", "Train", "label_data_2024-10-20_21-41-33.npz"),
            //Path.Combine("Datasets", "Labels", "Train", "label_data_2024-10-20_21-57-10.npz"),
            //Path.Combine("Datasets", "Labels", "Train", "label_data_2024-10-20_22-53-34.npz"),
            //Path.Combine("Datasets", "Labels", "Train", "label_data_2024-10-21_20-46-45.npz"),
            //Path.Combine("Datasets", "Labels", "Train", "label_data_2024-10-21_21-05-06.npz"),
            //Path.Combine("Datasets", "Labels", "Train", "label_data_2024-10-21_21-39-54.npz"),
        ];

        public override string[] LabelFileNpzIndex => [
            "arr_0",
            //"arr_0",
            //"arr_0",
            //"arr_0",
            //"label_data",
            //"arr_0",
            //"label_data",
            //"label_data",
            //"arr_0",
            //"label_data",
            //"label_data",
            //"arr_0",
            //"arr_0",
            //"arr_0",
            //"arr_0",
            //"arr_0",
            //"arr_0",
            //"arr_0",
            //"label_data",
            //"label_data",
        ];
    }
}
