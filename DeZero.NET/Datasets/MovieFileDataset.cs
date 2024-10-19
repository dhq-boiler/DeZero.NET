using DeZero.NET.Transforms;

namespace DeZero.NET.Datasets
{
    public abstract class MovieFileDataset : Dataset
    {
        public MovieFileDataset(bool train = true, Transform transform = null, Transform target_transform = null)
            : base(train, transform, target_transform)
        {
            PrepareLabelArray();
        }

        private void PrepareLabelArray()
        {
            var gpuIsEnabled = Gpu.Available && Gpu.Use;
            Gpu.Use = false;
            foreach (var labelfilePath in LabelFilePaths)
            {
                LabelArray = new NDarray[LabelFilePaths.Length];
                for (int i = 0; i < LabelFilePaths.Length; i++)
                {
                    var ndarray = xp.load(LabelFilePaths[i]);
                    LabelArray[i] = ndarray[LabelFileNpzIndex[i]].T;
                }
            }
            Gpu.Use = gpuIsEnabled;
        }

        public abstract string[] MovieFilePaths { get; }

        public abstract string[] LabelFilePaths { get; }

        public abstract string[] LabelFileNpzIndex { get; }

        public NDarray[] LabelArray { get; protected set; }

        public (NDarray, NDarray) this[int index1, int index2]
        {
            get
            {
                if (Label is null)
                {
                    return (Transform.Call<NDarray>(Data[index1][index2]), null);
                }
                else
                {
                    return (Transform.Call<NDarray>(Data[index1][index2]), TargetTransform.Call<NDarray>(Label[index1][index2]));
                }
            }
        }
    }
}
