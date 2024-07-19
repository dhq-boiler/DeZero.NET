using DeZero.NET.Core;
using DeZero.NET.Layers;

namespace DeZero.NET.Functions
{
    public class TimeDistributed : Function
    {
        public Layer Layer { get; }

        public TimeDistributed(Layer layer)
        {
            Layer = layer;
        }

        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>(0);

            int batch_size = x.Shape[0], time_steps = x.Shape[1];

            //入力をreshape
            var reshaped_x = x.reshape(batch_size * time_steps, -1);

            //レイヤーを適用
            var output = Layer.Call(reshaped_x)[0];

            return output.reshape(batch_size, time_steps, -1);
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Get<Variable>(0);

            int batch_size = gy.Shape[0], time_steps = gy.Shape[1];

            //勾配をreshape
            var reshaped_gy = gy.reshape(batch_size * time_steps, -1);

            //レイヤーの逆伝播
            var dx = Layer.Backward(reshaped_gy)[0];

            //勾配を元の形状に戻す
            return dx.reshape(batch_size, time_steps, -1);
        }
    }
}
