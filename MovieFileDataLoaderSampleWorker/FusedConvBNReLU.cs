using DeZero.NET;
using DeZero.NET.Core;
using DeZero.NET.Extensions;
using DeZero.NET.Layers;

namespace MovieFileDataLoaderSampleWorker;

public class FusedConvBNReLU : Layer, IDisposable
{
    private readonly Conv2dMobileNet _conv;
    private readonly NDarray _gamma;
    private readonly NDarray _beta;
    private readonly NDarray _runningMean;
    private readonly NDarray _runningVar;
    private readonly int _quantizationBits;
    private bool _disposed;
    private const float EPSILON = 1e-5f;

    public FusedConvBNReLU(int inChannels, int outChannels, int kernel,
        int stride, int padding, int quantizationBits)
    {
        _conv = new Conv2dMobileNet(outChannels, kernel, Dtype.float32,
            stride: stride, pad: padding, in_channels: inChannels);

        // Initialize BN parameters as NDarray to avoid Variable overhead
        var shape = new Shape(outChannels);
        _gamma = xp.ones(shape, dtype: Dtype.float32);
        _beta = xp.zeros(shape, dtype: Dtype.float32);
        _runningMean = xp.zeros(shape, dtype: Dtype.float32);
        _runningVar = xp.ones(shape, dtype: Dtype.float32);
        _quantizationBits = quantizationBits;
    }

    public override Variable[] Forward(params Variable[] inputs)
    {
        using var scope = new ComputationScope();
        var x = inputs[0];

        // Convolution
        using var convOutput = _conv.Forward(x)[0];
        var output = ApplyFusedBNReLU(convOutput);

        return new[] { output };
    }

    private Variable ApplyFusedBNReLU(Variable x)
    {
        using var scope = new ComputationScope();
        var data = x.Data.Value;
        var channels = data.shape[1];

        // Prepare broadcasting shapes for batch normalization
        var newShape = new[] { 1, channels, 1, 1 };
        using var reshapedGamma = _gamma.reshape(newShape);
        using var reshapedBeta = _beta.reshape(newShape);
        using var reshapedMean = _runningMean.reshape(newShape);
        using var reshapedVar = _runningVar.reshape(newShape);

        // Fused BN computation
        using var normalized = (data - reshapedMean) / xp.sqrt(reshapedVar + EPSILON);
        using var scaled = normalized * reshapedGamma + reshapedBeta;

        // ReLU
        using var zero = xp.array(0f);
        var activated = xp.maximum(zero, scaled);

        // Quantization
        var scale = (float)Math.Pow(2, _quantizationBits) - 1;
        using var min = activated.min();
        using var max = activated.max();
        using var eps = xp.array(float.Epsilon);
        using var normalized_q = (activated - min) / (max - min + eps);
        using var quantized = xp.round(normalized_q * scale) / scale;
        var rescaled = quantized * (max - min + eps) + min;

        return rescaled.ToVariable();
    }

    public void Dispose()
    {
        if (_disposed) return;
        _gamma?.Dispose();
        _beta?.Dispose();
        _runningMean?.Dispose();
        _runningVar?.Dispose();
        _conv?.Dispose();
        _disposed = true;
        GC.SuppressFinalize(this);
    }
}