using DeZero.NET;
using DeZero.NET.Layers.Recurrent;
using DeZero.NET.Models;
using L = DeZero.NET.Layers;

namespace MovieFileDataLoaderSampleWorker
{
    class DCNNModel : Model
    {
        public MobileNet Cnn { get; set; }
        public LSTM Lstm1 { get; set; }
        //public LSTM Lstm2 { get; set; }
        public L.Linear.Linear Fc1 { get; set; }
        public L.Linear.Linear Fc2 { get; set; }
        public L.Linear.Linear Fc3 { get; set; }
        private Variable lstm1H;
        private Variable lstm1C;
        private Variable lstm2H;
        private Variable lstm2C;

        public DCNNModel()
        {
            int mobilenet_output_channels = 960;
            int mobilenet_output_features = 960; // グローバル平均プーリング後は1x1になるため
            float width_mult = 0.75f;

            Cnn = new MobileNet(mobilenet_output_channels, width_mult);
            Lstm1 = new LSTM(512, in_size: 720);
            //Lstm2 = new LSTM(512, in_size: 512);
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
            try
            {
                var x = inputs[0];
                //Console.WriteLine($"DCNNModel Input shape: ({string.Join(", ", x.Shape.Dimensions)})");

                x = Cnn.Forward(x)[0];
                //Console.WriteLine($"MobileNet output shape: ({string.Join(", ", x.Shape.Dimensions)})");

                x = DeZero.NET.Functions.Reshape.Invoke(x, new Shape(x.Shape[0], 1, x.Shape[1]))[0];
                //Console.WriteLine($"Reshaped for LSTM input: ({string.Join(", ", x.Shape.Dimensions)})");

                try
                {
                    Variable lstm1Out;
                    (lstm1Out, lstm1H, lstm1C) = Lstm1.Forward(x, lstm1H, lstm1C);
                    //Console.WriteLine($"LSTM1 output shape: ({string.Join(", ", lstm1Out.Shape.Dimensions)})");

                    //Variable lstm2Out;
                    //(lstm2Out, lstm2H, lstm2C) = Lstm2.Forward(lstm1Out, lstm2H, lstm2C);
                    //Console.WriteLine($"LSTM2 output shape: ({string.Join(", ", lstm2Out.Shape.Dimensions)})");

                    // LSTMの出力をリシェイプ
                    lstm1Out = DeZero.NET.Functions.Reshape.Invoke(lstm1Out, new Shape(lstm1Out.Shape[0], -1))[0];
                    //lstm2Out = DeZero.NET.Functions.Reshape.Invoke(lstm2Out, new Shape(lstm2Out.Shape[0], -1))[0];
                    //Console.WriteLine($"Reshaped LSTM2 output: ({string.Join(", ", lstm2Out.Shape.Dimensions)})");

                    x = DeZero.NET.Functions.ReLU.Invoke(Fc1.Forward(lstm1Out)[0])[0];
                    //x = DeZero.NET.Functions.ReLU.Invoke(Fc1.Forward(lstm2Out)[0])[0];
                    //Console.WriteLine($"FC1 output shape: ({string.Join(", ", x.Shape.Dimensions)})");

                    x = DeZero.NET.Functions.ReLU.Invoke(Fc2.Forward(x)[0])[0];
                    //Console.WriteLine($"FC2 output shape: ({string.Join(", ", x.Shape.Dimensions)})");

                    x = Fc3.Forward(x)[0];
                    //Console.WriteLine($"Final output shape: ({string.Join(", ", x.Shape.Dimensions)})");

                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error in LSTM or FC layers: {ex.Message}");
                    Console.WriteLine($"Stack Trace: {ex.StackTrace}");
                }

                return new[] { x };
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error in Forward: {ex.Message}");
                Console.WriteLine($"Stack Trace: {ex.StackTrace}");
                throw;
            }
        }

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