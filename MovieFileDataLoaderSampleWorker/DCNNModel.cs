using DeZero.NET;
using DeZero.NET.Layers.Recurrent;
using DeZero.NET.Models;
using L = DeZero.NET.Layers;

namespace MovieFileDataLoaderSampleWorker
{
    class DCNNModel : Model
    {
        public ResNet18 Cnn { get; set; }
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
            Cnn = new ResNet18(resnet_output_size, Dtype.float32);
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
            var x = inputs[0];  // x shape: (batch_size, 3, 224, 224) where batch_size <= 20
            int batch_size = x.Shape[0];
            if (batch_size > 20)
            {
                throw new ArgumentException($"Expected batch size of 20 or less, but got {batch_size}");
            }

            x = Cnn.Forward(x)[0];  // x shape: (batch_size, 1000)
            x = DeZero.NET.Functions.Reshape.Invoke(x, new Shape(batch_size, 1, -1))[0];  // x shape: (batch_size, 1, 1000)

            // LSTM processing with state management
            Variable lstm1Out;
            (lstm1Out, lstm1H, lstm1C) = Lstm1.Forward(x, lstm1H, lstm1C);

            Variable lstm2Out;
            (lstm2Out, lstm2H, lstm2C) = Lstm2.Forward(lstm1Out, lstm2H, lstm2C);

            // Remove the time dimension without using Squeeze
            lstm2Out = DeZero.NET.Functions.Reshape.Invoke(lstm2Out, new Shape(batch_size, -1))[0];  // lstm2Out shape: (batch_size, 512)

            x = DeZero.NET.Functions.ReLU.Invoke(Fc1.Forward(lstm2Out)[0])[0];  // x shape: (batch_size, 256)
            x = DeZero.NET.Functions.ReLU.Invoke(Fc2.Forward(x)[0])[0];  // x shape: (batch_size, 128)
            x = Fc3.Forward(x)[0];  // x shape: (batch_size, 3)

            return new[] { x };
        }

        // Add a method to ensure LSTM states are properly initialized
        public void InitializeLSTMStates(int batch_size)
        {
            if (lstm1H == null || lstm1H.Shape[0] != batch_size)
            {
                lstm1H = new Variable(xp.zeros(new Shape(batch_size, 512), Dtype.float32));
                lstm1C = new Variable(xp.zeros(new Shape(batch_size, 512), Dtype.float32));
                lstm2H = new Variable(xp.zeros(new Shape(batch_size, 512), Dtype.float32));
                lstm2C = new Variable(xp.zeros(new Shape(batch_size, 512), Dtype.float32));
            }
        }
    }
}