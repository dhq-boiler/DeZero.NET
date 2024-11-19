using DeZero.NET;
using DeZero.NET.Core;
using DeZero.NET.Extensions;
using DeZero.NET.Layers;
using DeZero.NET.Log;

namespace MovieFileDataLoaderSampleWorker
{
    public class FusedConvBNReLU : Layer
    {
        private readonly float[] _fusedWeight;
        private readonly float[] _fusedBias;
        private readonly int _inChannels;
        private readonly int _outChannels;
        private readonly int _kernelSize;
        private readonly int _stride;
        private readonly int _padding;
        private readonly ILogger _logger;

        public FusedConvBNReLU(ILogger logger,
            Conv2dMobileNet conv,
            DeZero.NET.Layers.Normalization.BatchNorm bn,
            DeZero.NET.Layers.Activation.ReLU6 relu)
        {
            _inChannels = conv.InChannels.Value.Value;
            _outChannels = conv.OutChannels.Value;
            _kernelSize = conv.KernelSize.Value;
            _stride = conv.Stride.Value;
            _padding = conv.Pad.Value;
            _logger = logger;

            // 重みとバイアスの融合
            (_fusedWeight, _fusedBias) = FuseParameters(
                conv.W.Value.Data.Value.flatten().GetData<float[]>(),
                conv.b?.Value.Data.Value.flatten().GetData<float[]>(),
                bn.Gamma.Value.Data.Value.flatten().GetData<float[]>(),
                bn.Beta.Value.Data.Value.flatten().GetData<float[]>(),
                bn.AvgMean.Value.Data.Value.flatten().GetData<float[]>(),
                bn.AvgVar.Value.Data.Value.flatten().GetData<float[]>());
        }

        private (float[] weight, float[] bias) FuseParameters(
        float[] convWeight,
        float[] convBias,
        float[] gamma,
        float[] beta,
        float[] mean,
        float[] var)
        {
            var fusedWeight = new float[convWeight.Length];
            var fusedBias = new float[_outChannels];
            float eps = 1e-5f;

            for (int i = 0; i < _outChannels; i++)
            {
                float sqrt_var = MathF.Sqrt(var[i] + eps);
                float scale = gamma[i] / sqrt_var;

                // 重みの融合
                int weightsPerChannel = convWeight.Length / _outChannels;
                int startIdx = i * weightsPerChannel;
                int endIdx = startIdx + weightsPerChannel;

                for (int j = startIdx; j < endIdx; j++)
                {
                    fusedWeight[j] = convWeight[j] * scale;
                }

                // バイアスの融合
                fusedBias[i] = (convBias != null ? convBias[i] : 0f) * scale
                              - mean[i] * scale + beta[i];
            }

            return (fusedWeight, fusedBias);
        }

        public override Variable[] Forward(params Variable[] inputs)
        {
            var x = inputs[0];

            try
            {
                using (var scope = new ComputationScope())
                {
                    // 入力テンソルの形状を確認
                    _logger.LogDebug($"Input shape: {string.Join(", ", x.Shape.Dimensions)}");

                    // 重みテンソルの形状を構築
                    var weightShape = new Shape(_outChannels, _inChannels, _kernelSize, _kernelSize);
                    _logger.LogDebug($"Weight shape: {string.Join(", ", weightShape.Dimensions)}");

                    // 重みテンソルの変換
                    var weightArray = xp.array(_fusedWeight);
                    var reshapedWeight = weightArray.reshape(weightShape);
                    var weightVar = reshapedWeight.ToVariable();

                    // バイアステンソルの変換
                    var biasArray = xp.array(_fusedBias);
                    var biasVar = biasArray.ToVariable();

                    // Conv2Dの実行
                    var convOut = scope.Register(
                        DeZero.NET.Functions.Conv2d.Invoke(
                            x,
                            weightVar,
                            biasVar,
                            _stride,
                            _padding)[0]);

                    _logger.LogDebug($"Conv output shape: {string.Join(", ", convOut.Shape.Dimensions)}");

                    // ReLU6の適用
                    var result = DeZero.NET.Functions.Clip.Invoke(convOut, 0f, 6f)[0];
                    _logger.LogDebug($"Final output shape: {string.Join(", ", result.Shape.Dimensions)}");

                    return new[] { result };
                }
            }
            catch (Exception ex)
            {
                _logger.LogError($"Forward pass error in FusedConvBNReLU: {ex.Message}");
                _logger.LogDebug($"Input channels: {_inChannels}, Output channels: {_outChannels}");
                _logger.LogDebug($"Kernel size: {_kernelSize}, Stride: {_stride}, Padding: {_padding}");
                throw;
            }
        }

        public void LogLayerInfo()
        {
            _logger.LogDebug($"Layer configuration:");
            _logger.LogDebug($"Input channels: {_inChannels}");
            _logger.LogDebug($"Output channels: {_outChannels}");
            _logger.LogDebug($"Kernel size: {_kernelSize}");
            _logger.LogDebug($"Stride: {_stride}");
            _logger.LogDebug($"Padding: {_padding}");
            _logger.LogDebug($"Fused weight length: {_fusedWeight.Length}");
            _logger.LogDebug($"Fused bias length: {_fusedBias.Length}");
        }
    }
}
