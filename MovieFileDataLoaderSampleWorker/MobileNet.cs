
using DeZero.NET;
using DeZero.NET.Core;
using DeZero.NET.Layers;
using DeZero.NET.Log;
using DeZero.NET.Models;
using System.Collections.ObjectModel;

namespace MovieFileDataLoaderSampleWorker
{
    public class MobileNet : Model
    {
        public Property<ObservableCollection<Layer>> _layers { get; } = new(nameof(_layers));

        private int i = 0;
        private readonly ILogger _logger;

        public MobileNet(ILogger logger, int num_classes = 1000, float width_mult = 1.0f)
        {
            RegisterEvent(_layers);

            _layers.Value = new ObservableCollection<Layer>();
            _logger = logger;

            // 初期レイヤー：3チャンネル（RGB）入力から24チャンネル出力
            AddConvBNReLU(3, 24, stride: 2, width_mult: width_mult);

            // Inverted Residual Blocksの追加（チャンネル数を調整）
            AddInvertedResidual(24, 24, 1, 1, width_mult);
            AddInvertedResidual(24, 24, 2, 6, width_mult);
            AddInvertedResidual(24, 32, 1, 6, width_mult);
            AddInvertedResidual(32, 32, 2, 6, width_mult);
            AddInvertedResidual(32, 32, 1, 6, width_mult);
            AddInvertedResidual(32, 64, 1, 6, width_mult);
            AddInvertedResidual(64, 64, 2, 6, width_mult);
            AddInvertedResidual(64, 64, 1, 6, width_mult);
            AddInvertedResidual(64, 96, 1, 6, width_mult);
            AddInvertedResidual(96, 96, 1, 6, width_mult);
            AddInvertedResidual(96, 96, 1, 6, width_mult);
            AddInvertedResidual(96, 160, 2, 6, width_mult);
            AddInvertedResidual(160, 160, 1, 6, width_mult);
            AddInvertedResidual(160, 320, 1, 6, width_mult);

            // 最後の畳み込み層
            int last_channel = (int)Math.Round(320 * width_mult);
            AddConvBNReLU(last_channel, num_classes, kernel: 1, stride: 1, padding: 0, width_mult: 1);

            // グローバル平均プーリングは Forward メソッドで動的に追加します
        }

        private void AddConvBNReLU(int in_channels, int out_channels, int kernel = 3, int stride = 1, int padding = 1, float width_mult = 1.0f)
        {
            int oc = (int)Math.Round(out_channels * width_mult);
            _layers.Value.Add(new Conv2dMobileNet(oc, kernel, Dtype.float32, stride: stride, pad: padding, in_channels: in_channels));
            SetAttribute($"Conv2d-{i++}", _layers.Value[^1]);

            // BatchNorm の初期化を修正
            var shape = new Shape(oc);
            var mean = new Variable(xp.zeros(shape, dtype: Dtype.float32));
            var var = new Variable(xp.ones(shape, dtype: Dtype.float32));
            var gamma = new Variable(xp.ones(shape, dtype: Dtype.float32));
            var beta = new Variable(xp.zeros(shape, dtype: Dtype.float32));

            var batchNorm = new DeZero.NET.Layers.Normalization.BatchNorm(oc);
            batchNorm.InitParams(mean, var, gamma, beta);
            _layers.Value.Add(batchNorm);
            SetAttribute($"BatchNorm-{i++}", _layers.Value[^1]);

            _layers.Value.Add(new DeZero.NET.Layers.Activation.ReLU6());
            SetAttribute($"ReLU6-{i++}", _layers.Value[^1]);
        }

        private void AddInvertedResidual(int in_channels, int out_channels, int stride, int expand_ratio, float width_mult)
        {
            int ic = (int)Math.Round(in_channels * width_mult);
            int oc = (int)Math.Round(out_channels * width_mult);
            int hidden_dim = ic * expand_ratio;

            var layers = new ObservableCollection<Layer>();

            if (expand_ratio != 1)
            {
                layers.Add(new Conv2dMobileNet(hidden_dim, 1, Dtype.float32, in_channels: ic));
                SetAttribute($"Conv2d-{i++}", layers[^1]);

                // BatchNorm の初期化を修正
                var shape = new Shape(hidden_dim);
                var mean = new Variable(xp.zeros(shape, dtype: Dtype.float32));
                var var = new Variable(xp.ones(shape, dtype: Dtype.float32));
                var gamma = new Variable(xp.ones(shape, dtype: Dtype.float32));
                var beta = new Variable(xp.zeros(shape, dtype: Dtype.float32));

                var batchNorm = new DeZero.NET.Layers.Normalization.BatchNorm(hidden_dim);
                batchNorm.InitParams(mean, var, gamma, beta);
                layers.Add(batchNorm);
                SetAttribute($"BatchNorm-{i++}", layers[^1]);

                layers.Add(new DeZero.NET.Layers.Activation.ReLU6());
                SetAttribute($"ReLU6-{i++}", layers[^1]);
            }

            // Depthwise convolution の BatchNorm
            layers.Add(new Conv2dMobileNet(hidden_dim, 3, Dtype.float32, stride: stride, pad: 1, in_channels: hidden_dim));
            SetAttribute($"Conv2d-{i++}", layers[^1]);

            {
                var shape = new Shape(hidden_dim);
                var mean = new Variable(xp.zeros(shape, dtype: Dtype.float32));
                var var = new Variable(xp.ones(shape, dtype: Dtype.float32));
                var gamma = new Variable(xp.ones(shape, dtype: Dtype.float32));
                var beta = new Variable(xp.zeros(shape, dtype: Dtype.float32));

                var batchNorm = new DeZero.NET.Layers.Normalization.BatchNorm(hidden_dim);
                batchNorm.InitParams(mean, var, gamma, beta);
                layers.Add(batchNorm);
                SetAttribute($"BatchNorm-{i++}", layers[^1]);
            }

            layers.Add(new DeZero.NET.Layers.Activation.ReLU6());
            SetAttribute($"ReLU6-{i++}", layers[^1]);

            // Pointwise convolution の BatchNorm
            layers.Add(new Conv2dMobileNet(oc, 1, Dtype.float32, in_channels: hidden_dim));
            SetAttribute($"Conv2d-{i++}", layers[^1]);

            {
                var shape = new Shape(oc);
                var mean = new Variable(xp.zeros(shape, dtype: Dtype.float32));
                var var = new Variable(xp.ones(shape, dtype: Dtype.float32));
                var gamma = new Variable(xp.ones(shape, dtype: Dtype.float32));
                var beta = new Variable(xp.zeros(shape, dtype: Dtype.float32));

                var batchNorm = new DeZero.NET.Layers.Normalization.BatchNorm(oc);
                batchNorm.InitParams(mean, var, gamma, beta);
                layers.Add(batchNorm);
                SetAttribute($"BatchNorm-{i++}", layers[^1]);
            }

            // 残りのコードは同じ
            if (ic == oc && stride == 1)
            {
                _layers.Value.Add(new InvertedResidualBlock(layers, ic));
                SetAttribute($"InvertedResidualBlock-{i++}", _layers.Value[^1]);
            }
            else
            {
                if (stride != 1)
                {
                    layers.Add(new Conv2dMobileNet(oc, 1, Dtype.float32, stride: stride, in_channels: ic));
                    SetAttribute($"Conv2d-{i++}", layers[^1]);

                    var shape = new Shape(oc);
                    var mean = new Variable(xp.zeros(shape, dtype: Dtype.float32));
                    var var = new Variable(xp.ones(shape, dtype: Dtype.float32));
                    var gamma = new Variable(xp.ones(shape, dtype: Dtype.float32));
                    var beta = new Variable(xp.zeros(shape, dtype: Dtype.float32));

                    var batchNorm = new DeZero.NET.Layers.Normalization.BatchNorm(oc);
                    batchNorm.InitParams(mean, var, gamma, beta);
                    layers.Add(batchNorm);
                    SetAttribute($"BatchNorm-{i++}", layers[^1]);
                }
                layers.ToList().ForEach(layer => _layers.Value.Add(layer));
            }
        }


        public override Variable[] Forward(params Variable[] inputs)
        {
            var x = inputs[0];
            for (int i = 0; i < _layers.Value.Count; i++)
            {
                var layer = _layers.Value[i];
                // グローバル平均プーリングを動的に追加
                if (i == _layers.Value.Count - 2) // 分類層の直前
                {
                    var poolSize = (x.Shape[2], x.Shape[3]);
                    var avgPooling = new DeZero.NET.Layers.Convolution.AveragePooling(poolSize, stride: 1, pad: 0);
                    SetAttribute($"AveragePooling-{this.i++}", avgPooling);
                    x = avgPooling.Forward(x)[0];
                }
                x = layer.Forward(x)[0];
                //Console.WriteLine($"After Layer {i} ({layer.GetType().Name}): {string.Join(", ", x.Shape.Dimensions)}");
            }
            //Console.WriteLine($"MobileNet final output shape: ({string.Join(", ", x.Shape.Dimensions)})");
            return new[] { x };
        }
    }

    public class InvertedResidualBlock : Layer
    {
        public Property<ObservableCollection<Layer>> _layers { get; } = new(nameof(_layers));
        public Property<int> _in_channels { get; } = new(nameof(_in_channels));

        public InvertedResidualBlock(ObservableCollection<Layer> layers, int in_channels)
        {
            RegisterEvent(_layers, _in_channels);

            _layers.Value = layers;
            _in_channels.Value = in_channels;
        }

        public override Variable[] Forward(params Variable[] inputs)
        {
            var x = inputs[0];
            using var identity = x.copy();

            foreach (var layer in _layers.Value)
            {
                using var _x = layer.Forward(x)[0];
                if (!ReferenceEquals(_layers.Value.First(), layer))
                {
                    x.Dispose();
                }
                x = _x.copy();
            }

            using var __x = DeZero.NET.Functions.Add.Invoke(x, identity).Item1[0];
            x.Dispose();
            return new[] { __x.copy() };
        }
    }
}