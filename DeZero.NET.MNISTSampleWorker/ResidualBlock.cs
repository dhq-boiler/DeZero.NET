using L = DeZero.NET.Layers;

namespace DeZero.NET.MNISTSampleWorker
{
    public class ResidualBlock : Models.Sequential
    {
        public ResidualBlock(int inChannels, int outChannels, int stride = 1) : base(ConstructModel(inChannels, outChannels, stride).ToArray())
        {
            
        }

        private static List<L.Layer> ConstructModel(int inChannels, int outChannels, int stride)
        {
            var layers = new List<L.Layer>()
            {
                new L.Convolution.Conv2dResNet("a", outChannels, 3, Dtype.float32, stride: 1, pad: 1),
                new L.Normalization.BatchNorm(),
                new L.Activation.ReLU(),
                new L.Convolution.Conv2dResNet("b", outChannels, 3, Dtype.float32, stride: 1, pad: 1),
                new L.Normalization.BatchNorm(),
                new L.Activation.ReLU(),
                new L.Convolution.Conv2dResNet("c", outChannels, 3, Dtype.float32, stride: 1, pad: 1),
                new L.Normalization.BatchNorm(),
            };

            if (inChannels != outChannels)
            {
                var shortcut = new Models.Sequential(new L.Layer[]
                {
                    new L.Convolution.Conv2dResNet("d", outChannels, 3, Dtype.float32, stride: 1, pad: 1),
                    new L.Normalization.BatchNorm()
                });
                layers.Add(new L.SkipConnection(shortcut));
            }
            else
            {
                layers.Add(new L.Projection(xs => xs));
            }
            layers.Add(new L.Activation.ReLU());
            return layers;
        }
    }
}
