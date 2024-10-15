using DeZero.NET;
using DeZero.NET.Layers.Recurrent;
using DeZero.NET.Models;
using DeZero.NET.OpenCv;
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

        private Variable lstm1H;
        private Variable lstm1C;
        private Variable lstm2H;
        private Variable lstm2C;

        public DCNNModel()
        {
            int resnet_output_size = 1000;
            Cnn = new ResNet50(Dtype.float32);
            Lstm1 = new LSTM(512, in_size: resnet_output_size);
            Lstm2 = new LSTM(512, in_size: 512);
            Fc1 = new L.Linear.Linear(256, in_size: 512);
            Fc2 = new L.Linear.Linear(128, in_size: 256);
            Fc3 = new L.Linear.Linear(3, in_size: 128);

            ResetState();
        }

        public void ResetState()
        {
            lstm1H = null;
            lstm1C = null;
            lstm2H = null;
            lstm2C = null;
        }

        public override Variable[] Forward(params Variable[] inputs)
        {
            var x = inputs[0];  // Assume x is a single frame

            // Resize for CNN input (224x224)
            x.Data.Value = Cv2.resize(x.Data.Value, (224, 224), Cv2.INTER_LINEAR);
            x.Data.Value.resize(new Shape(1, 3, 224, 224));

            // CNN feature extraction
            x = Cnn.Forward(x)[0];

            // Reshape for LSTM input
            x = DeZero.NET.Functions.Reshape.Invoke(x, new Shape(1, -1))[0];

            // LSTM processing
            Variable h1, h2;
            (lstm1H, lstm1C) = Lstm1.Forward(x, lstm1H, lstm1C);
            (lstm2H, lstm2C) = Lstm2.Forward(lstm1H, lstm2H, lstm2C);

            // Fully connected layers
            x = DeZero.NET.Functions.ReLU.Invoke(Fc1.Forward(lstm2H)[0])[0];
            x = DeZero.NET.Functions.ReLU.Invoke(Fc2.Forward(x)[0])[0];
            x = Fc3.Forward(x)[0];

            // Reshape for output : (3)
            x = DeZero.NET.Functions.Reshape.Invoke(x, new Shape(3))[0];

            return [x];
        }
    }
}