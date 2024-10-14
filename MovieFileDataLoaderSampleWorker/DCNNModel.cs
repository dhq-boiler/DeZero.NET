using DeZero.NET;
using DeZero.NET.Layers.Recurrent;
using DeZero.NET.Models;
using L = DeZero.NET.Layers;

namespace MovieFileDataLoaderSampleWorker
{
    class DCNNModel : Model
    {
        public ResNet50 Cnn { get; set; }
        public LSTM Lstm1 { get; set; }
        public LSTM Lstm2 { get; set; }
        public L.Linear.Linear Fc1 { get; set; }
        public L.Linear.Linear Fc2 { get; set; }
        public L.Linear.Linear Fc3 { get; set; }

        public DCNNModel()
        {
            // ResNet50の出力チャンネル数は2048
            int resnet_output_size = 2048;

            Cnn = new ResNet50(Dtype.float32);
            Lstm1 = new LSTM(512, in_size: resnet_output_size);
            Lstm2 = new LSTM(512, in_size: 512);
            Fc1 = new L.Linear.Linear(256, in_size: 512);
            Fc2 = new L.Linear.Linear(128, in_size: 256);
            Fc3 = new L.Linear.Linear(3, in_size: 128);
        }

        public override Variable[] Forward(params Variable[] inputs)
        {
            var x = inputs[0];
            var batchSize = x.Shape[0];
            var nFrames = x.Shape[1];

            // Reshape for CNN input
            x = DeZero.NET.Functions.Reshape.Invoke(x[0], new Shape(batchSize * nFrames, 3, 224, 224))[0];

            // CNN feature extraction
            x = Cnn.Forward(x)[0];

            // Reshape for LSTM input
            x = DeZero.NET.Functions.Reshape.Invoke(x[0], new Shape(batchSize, nFrames, -1))[0];

            // LSTM processing
            for (int i = 0; i < nFrames; i++)
            {
                var frame = DeZero.NET.Functions.GetItem.Invoke(x, new NDarray(0), new NDarray(x.__len__));
                var h1 = Lstm1.Forward(frame);
                x = Lstm2.Forward(h1)[0];
            }

            // Fully connected layers
            x = DeZero.NET.Functions.ReLU.Invoke(Fc1.Forward(x)[0])[0];
            x = DeZero.NET.Functions.ReLU.Invoke(Fc2.Forward(x)[0])[0];
            x = Fc3.Forward(x)[0];

            return [x];
        }
    }
}
