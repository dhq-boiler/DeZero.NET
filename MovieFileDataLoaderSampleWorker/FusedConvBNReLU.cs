using DeZero.NET;
using DeZero.NET.Core;
using DeZero.NET.Extensions;
using DeZero.NET.Layers;
using DocumentFormat.OpenXml.Wordprocessing;

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
        //using var scope = new ComputationScope();
        var x = inputs[0];

        // Convolution
        using var convOutput = _conv.Forward(x)[0];
        using var output = ApplyFusedBNReLU(convOutput);

        return new[] { output.copy() };
    }

    private Variable ApplyFusedBNReLU(Variable x)
    {
        //using var scope = new ComputationScope();
        var data = x.Data.Value;
        using var data_shape = data.shape;
        var channels = data_shape[1];

        // Prepare broadcasting shapes for batch normalization
        var newShape = new[] { 1, channels, 1, 1 };
        using var reshapedGamma = _gamma.reshape(newShape);
        using var reshapedBeta = _beta.reshape(newShape);
        using var reshapedMean = _runningMean.reshape(newShape);
        using var reshapedVar = _runningVar.reshape(newShape);

        // Fused BN computation
        using var a = data - reshapedMean;
        using var b = reshapedVar + EPSILON;
        using var c = xp.sqrt(b);
        using var normalized = a / c;
        using var d = normalized * reshapedGamma;
        using var scaled = d + reshapedBeta;

        // ReLU
        using var zero = xp.array(0f);
        using var activated = xp.maximum(zero, scaled);

        // Quantization
        var scale = (float)Math.Pow(2, _quantizationBits) - 1;
        using var min = activated.min();
        using var max = activated.max();
        using var eps = xp.array(float.Epsilon);
        using var e = activated - min;
        using var f = max - min;
        using var g = f + eps;
        using var normalized_q = e / g;
        using var h = normalized_q * scale;
        using var i = h.round();
        using var quantized = i / scale;
        using var j = quantized * g;
        var rescaled = j + min;

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