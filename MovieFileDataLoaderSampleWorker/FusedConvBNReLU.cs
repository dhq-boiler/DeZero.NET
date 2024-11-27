using DeZero.NET;
using DeZero.NET.Core;
using DeZero.NET.Extensions;
using DeZero.NET.Functions;
using DeZero.NET.Models;

namespace MovieFileDataLoaderSampleWorker;

public class FusedConvBNReLU : Model, IDisposable
{
    private Property<Conv2dMobileNet> _conv { get; } = new Property<Conv2dMobileNet>(nameof(_conv));
    private Property<Variable> _gamma { get; } = new Property<Variable>(nameof(_gamma));
    private Property<Variable> _beta { get; } = new Property<Variable>(nameof(_beta));
    private Property<Variable> _runningMean { get; } = new Property<Variable>(nameof(_runningMean));
    private Property<Variable> _runningVar { get; } = new Property<Variable>(nameof(_runningVar));
    private Property<int> _quantizationBits { get; } = new Property<int>(nameof(_quantizationBits));
    private bool _disposed;
    private const float EPSILON = 1e-5f;

    public FusedConvBNReLU(int inChannels, int outChannels, int kernel,
        int stride, int padding, int quantizationBits)
    {
        _conv.Value = new Conv2dMobileNet(outChannels, kernel, Dtype.float32,
            stride: stride, pad: padding, in_channels: inChannels);
        SetAttribute("Conv2dMobileNet", _conv.Value);

        // Initialize BN parameters as NDarray to avoid Variable overhead
        var shape = new Shape(outChannels);
        _gamma.Value = xp.ones(shape, dtype: Dtype.float32).ToVariable();
        _beta.Value = xp.zeros(shape, dtype: Dtype.float32).ToVariable();
        _runningMean.Value = xp.zeros(shape, dtype: Dtype.float32).ToVariable();
        _runningVar.Value = xp.ones(shape, dtype: Dtype.float32).ToVariable();
        _quantizationBits.Value = quantizationBits;

        SetAttribute("gamma", _gamma.Value);
        SetAttribute("beta", _beta.Value);
        SetAttribute("runningMean", _runningMean.Value);
        SetAttribute("runningVar", _runningVar.Value);

        RegisterEvent(_conv, _gamma, _beta, _runningMean, _runningVar, _quantizationBits);
    }

    public override Variable[] Forward(params Variable[] inputs)
    {
        //using var scope = new ComputationScope();
        var x = inputs[0];

        // Convolution
        using var convOutput = _conv.Value.Forward(x)[0];
        using var output = ApplyFusedBNReLU(convOutput);

        return new[] { output.copy() };
    }

    private Variable ApplyFusedBNReLU(Variable x)
    {
        //using var scope = new ComputationScope();
        var data = x;
        using var data_shape = data.Shape;
        var channels = data_shape[1];

        // Prepare broadcasting shapes for batch normalization
        var newShape = new[] { 1, channels, 1, 1 };
        using var reshapedGamma = DeZero.NET.Functions.Reshape.Invoke(_gamma.Value, newShape)[0];
        using var reshapedBeta = DeZero.NET.Functions.Reshape.Invoke(_beta.Value, newShape)[0];
        using var reshapedMean = DeZero.NET.Functions.Reshape.Invoke(_runningMean.Value, newShape)[0];
        using var reshapedVar = DeZero.NET.Functions.Reshape.Invoke(_runningVar.Value, newShape)[0];

        // Fused BN computation
        using var epsilon = new NDarray(EPSILON).ToVariable();
        using var a = Subtract.Invoke(data, reshapedMean)[0];
        using var b = Add.Invoke(reshapedVar, epsilon).Item1[0];
        using var c = Sqrt.Invoke(b)[0];
        using var normalized = Div.Invoke(a, c)[0];
        using var d = Mul.Invoke(normalized, reshapedGamma)[0];
        using var scaled = Add.Invoke(d, reshapedBeta).Item1[0];

        // ReLU
        using var zero = xp.array(0f).ToVariable();
        using var activated = Maximum.Invoke(zero, scaled).Item1[0];

        // Quantization
        using var scale = new NDarray((float)Math.Pow(2, _quantizationBits.Value) - 1).ToVariable();
        using var min = activated.Data.Value.min().ToVariable();
        using var max = activated.Data.Value.max().ToVariable();
        using var eps = xp.array(float.Epsilon).ToVariable();
        using var e = Subtract.Invoke(activated, min)[0];
        using var f = Subtract.Invoke(max, min)[0];
        using var g = Add.Invoke(f, eps).Item1[0];
        using var normalized_q = Div.Invoke(e, g)[0];
        using var h = Mul.Invoke(normalized_q, scale)[0];
        using var i = Round.Invoke(h)[0];
        using var quantized = Div.Invoke(i, scale)[0];
        using var j = Mul.Invoke(quantized, g)[0];
        using var rescaled = Add.Invoke(j, min).Item1[0];

        return rescaled.copy();
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