using L = DeZero.NET.Layers;

namespace DeZero.NET.MNISTSampleWorker
{
    public class ResNet : Models.Sequential
    {
        public ResNet(int[] numBlocks, int numClasses = 10) : base(ConstructModel(numBlocks, numClasses).ToArray())
        {
        }

        private static List<L.Layer> ConstructModel(int[] numBlocks, int numClasses)
        {
            var layers = new List<L.Layer>()
            {
                new L.Convolution.Conv2dResNet("e", 16, 3, Dtype.float32, stride: 1, pad: 1),
                new L.Normalization.BatchNorm(),
                new L.Activation.ReLU()
            };

            int inChannels = 16;
            int[] outChannels = { 16, 32, 64 };
            int[] strides = { 1, 1, 1 };

            for (int i = 0; i < 3; i++)
            {
                layers.AddRange(MakeLayer(inChannels, outChannels[i], numBlocks[i], strides[i]));
                inChannels = outChannels[i];
            }

            layers.AddRange(new L.Layer[]
            {
                new L.Convolution.GlobalAveragePooling(),
                new L.Linear.Linear(numClasses),
                new L.Activation.Softmax()
            });
            return layers;
        }

        private static L.Layer[] MakeLayer(int inChannels, int outChannels, int numBlocks, int stride)
        {
            var layers = new List<L.Layer>();

            layers.Add(new ResidualBlock(inChannels, outChannels, stride));

            for (int i = 1; i < numBlocks; i++)
            {
                layers.Add(new ResidualBlock(outChannels, outChannels, 1));
            }

            return layers.ToArray();
        }
    }
}
