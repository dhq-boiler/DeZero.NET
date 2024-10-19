using DeZero.NET;
using DeZero.NET.Layers.Recurrent;
using DeZero.NET.Models;
using L = DeZero.NET.Layers;

namespace MovieFileDataLoaderSampleWorker
{
    class DCNNModel : Model
    {
        public MobileNet Cnn { get; set; }
        //public LSTM Lstm1 { get; set; }
        //public LSTM Lstm2 { get; set; }
        public GRU Gru1 { get; set; }
        public L.Linear.Linear Fc1 { get; set; }
        public L.Linear.Linear Fc2 { get; set; }
        public L.Linear.Linear Fc3 { get; set; }
        //private Variable lstm1H;
        //private Variable lstm1C;
        //private Variable lstm2H;
        //private Variable lstm2C;

        public DCNNModel()
        {
            int mobilenet_output_channels = 68; // MobileNetの実際の出力チャンネル数
            float width_mult = 0.75f;
            Cnn = new MobileNet(mobilenet_output_channels, width_mult);
            int gru_input_size = mobilenet_output_channels * 1 * 1;
            Gru1 = new GRU(gru_input_size, 512);
            Fc1 = new L.Linear.Linear(256, in_size: 512);
            Fc2 = new L.Linear.Linear(128, in_size: 256);
            Fc3 = new L.Linear.Linear(3, in_size: 128);
            SetAttribute("MobileNet", Cnn);
            SetAttribute("GRU1", Gru1);
            SetAttribute("Fc1", Fc1);
            SetAttribute("Fc2", Fc2);
            SetAttribute("Fc3", Fc3);
            ResetState();
        }

        public void ResetState()
        {
            Gru1.ResetState();
            //lstm1H = null;
            //lstm1C = null;
            //lstm2H = null;
            //lstm2C = null;
        }

        public override Variable[] Forward(params Variable[] inputs)
        {
            var x = inputs[0];
            //Console.WriteLine($"Input shape: ({string.Join(", ", x.Shape.Dimensions)})");

            x = Cnn.Forward(x)[0];
            //Console.WriteLine($"CNN output shape: ({string.Join(", ", x.Shape.Dimensions)})");

            // CNNの出力を (バッチサイズ, 特徴量) にリシェイプ
            x = DeZero.NET.Functions.Reshape.Invoke(x, new Shape(x.Shape[0], -1))[0];
            //Console.WriteLine($"Reshaped CNN output: ({string.Join(", ", x.Shape.Dimensions)})");

            // GRUへの入力は (バッチサイズ, 特徴量) のままにする
            var gru1Out = Gru1.Forward(x)[0];
            //Console.WriteLine($"GRU output shape: ({string.Join(", ", gru1Out.Shape.Dimensions)})");

            // GRUの出力が3次元の場合、2次元に変換
            if (gru1Out.Shape.Dimensions.Length == 3)
            {
                gru1Out = DeZero.NET.Functions.Reshape.Invoke(gru1Out, new Shape(gru1Out.Shape[0], -1))[0];
                //Console.WriteLine($"Reshaped GRU output: ({string.Join(", ", gru1Out.Shape.Dimensions)})");
            }

            x = DeZero.NET.Functions.ReLU.Invoke(Fc1.Forward(gru1Out)[0])[0];
            x = DeZero.NET.Functions.ReLU.Invoke(Fc2.Forward(x)[0])[0];
            x = Fc3.Forward(x)[0];
            //Console.WriteLine($"Final output shape: ({string.Join(", ", x.Shape.Dimensions)})");

            return new[] { x };
        }

        public void InitializeLSTMStates(int batch_size)
        {
            //if (lstm1H == null || lstm1H.Shape[0] != batch_size)
            //{
            //    lstm1H = new Variable(xp.zeros(new Shape(batch_size, 512), Dtype.float32));
            //    lstm1C = new Variable(xp.zeros(new Shape(batch_size, 512), Dtype.float32));
            //    lstm2H = new Variable(xp.zeros(new Shape(batch_size, 512), Dtype.float32));
            //    lstm2C = new Variable(xp.zeros(new Shape(batch_size, 512), Dtype.float32));
            //}
            Gru1.ResetState();
        }
    }
}