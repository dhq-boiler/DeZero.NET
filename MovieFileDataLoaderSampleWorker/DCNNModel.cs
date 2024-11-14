using DeZero.NET;
using DeZero.NET.Layers.Recurrent;
using DeZero.NET.Models;
using L = DeZero.NET.Layers;

namespace MovieFileDataLoaderSampleWorker
{
    class DCNNModel : Model
    {
        public MobileNet Cnn { get; set; }
        public GRU Gru1 { get; set; }
        public L.Linear.Linear Fc1 { get; set; }
        public L.Linear.Linear Fc2 { get; set; }
        public L.Linear.Linear Fc3 { get; set; }

        private const int SEQUENCE_LENGTH = 60; // 適切な長さに調整可能
        private Queue<Variable> stateQueue;

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

            stateQueue = new Queue<Variable>();
            ResetState();
        }

        public void ResetState()
        {
            Gru1.ResetState();
            stateQueue.Clear();
        }

        public override Variable[] Forward(params Variable[] inputs)
        {
            var x = inputs[0];
            x = Cnn.Forward(x)[0];
            x = DeZero.NET.Functions.Reshape.Invoke(x, new Shape(x.Shape[0], -1))[0];

            // シーケンス管理による最適化
            var gru1Out = Gru1.Forward(x)[0];
            stateQueue.Enqueue(gru1Out);

            if (stateQueue.Count > SEQUENCE_LENGTH)
            {
                stateQueue.Dequeue();
                // 定期的に状態をリセットして再計算
                if (stateQueue.Count == SEQUENCE_LENGTH)
                {
                    Gru1.ResetState();
                    foreach (var state in stateQueue)
                    {
                        Gru1.Forward(state);
                    }
                }
            }

            if (gru1Out.Shape.Dimensions.Length == 3)
            {
                gru1Out = DeZero.NET.Functions.Reshape.Invoke(gru1Out, new Shape(gru1Out.Shape[0], -1))[0];
            }

            x = DeZero.NET.Functions.ReLU.Invoke(Fc1.Forward(gru1Out)[0])[0];
            x = DeZero.NET.Functions.ReLU.Invoke(Fc2.Forward(x)[0])[0];
            x = Fc3.Forward(x)[0];

            return [x];
        }

        public void InitializeLSTMStates(int batch_size)
        {
            ResetState();
        }
    }
}