using DeZero.NET;
using DeZero.NET.Datasets;
using DeZero.NET.OpenCv;

namespace MovieFileDataLoaderSampleWorker
{
    internal class SampleDataLoader : MovieFileDataLoader
    {
        public SampleDataLoader(MovieFileDataset dataset, int batchSize, Action changeMovieAction, bool shuffle = true, int bufferSize = 180, int prefetchBatches = 3) : base(dataset, batchSize, changeMovieAction, shuffle)
        {
        }

        public override NDarray ProcessFrame(NDarray frame)
        {
            // フレーム処理の最適化
            var ndArray = frame.reshape(1080, 1920 * 3);
            ndArray = Cv2.resize(ndArray, (224, 224 * 3), Cv2.INTER_LANCZOS4);
            ndArray = ndArray.reshape(3, 224, 224);
            return ndArray;
        }
    }
}
