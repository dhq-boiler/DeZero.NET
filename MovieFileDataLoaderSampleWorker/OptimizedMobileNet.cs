using DeZero.NET;
using DeZero.NET.Core;
using DeZero.NET.Extensions;
using DeZero.NET.Layers;
using DeZero.NET.Log;
using DeZero.NET.Models;
using Python.Runtime;
using System.Collections.ObjectModel;

namespace MovieFileDataLoaderSampleWorker
{
    public class OptimizedMobileNet : Model, IDisposable
    {
        private readonly int _quantizationBits;
        private readonly ILogger _logger;
        private readonly bool _enableFusedOperations;
        private readonly int _batchSize;
        private readonly MemoryManager _memoryManager;
        public Property<List<Layer>> _layers = new(nameof(_layers));
        public IReadOnlyList<Layer> Layers => _layers.Value;
        private int _layerIndex = 0;
        private bool _disposed;

        public OptimizedMobileNet(
            ILogger logger,
            int numClasses = 1000,
            float widthMult = 1.0f,
            int cacheSize = 100,
            int quantizationBits = 8,
            bool enableFusedOperations = true,
            int batchSize = 32)
        {
            _logger = logger;
            _quantizationBits = quantizationBits;
            _enableFusedOperations = enableFusedOperations;
            _batchSize = batchSize;
            _memoryManager = new MemoryManager(logger, 85);
            _layers.Value = new List<Layer>();

            InitializeLayers(numClasses, widthMult);
            RegisterEvent(_layers);
        }

        private void InitializeLayers(int numClasses, float widthMult)
        {
            using var scope = new ComputationScope();
            int[] channels = { 24, 32, 32, 32 };
            int[] repeats = { 2, 2, 2, 1 };

            AddOptimizedConvBNReLU(3, channels[0], stride: 2, widthMult: widthMult);

            int inChannels = channels[0];
            for (int i = 0; i < channels.Length; i++)
            {
                for (int j = 0; j < repeats[i]; j++)
                {
                    int stride = (j == 0 && i > 0) ? 2 : 1;
                    AddOptimizedInvertedResidual(inChannels, channels[i], stride, widthMult);
                    inChannels = channels[i];
                }
            }

            int lastChannel = 32;
            AddOptimizedConvBNReLU(lastChannel, 32, kernel: 1, stride: 1, padding: 0);
        }

        private void AddOptimizedConvBNReLU(int inChannels, int outChannels, int kernel = 3,
            int stride = 1, int padding = 1, float widthMult = 1.0f)
        {
            if (_enableFusedOperations)
            {
                // Conv+BN+ReLUを1つの操作にフュージョン
                AddFusedConvBNReLU(inChannels, outChannels, kernel, stride, padding, widthMult);
            }
            else
            {
                // 従来の実装
                AddConvBNReLU(inChannels, outChannels, kernel, stride, padding, widthMult);
            }
        }

        private void AddFusedConvBNReLU(int inChannels, int outChannels, int kernel,
            int stride, int padding, float widthMult)
        {
            int oc = (int)(outChannels * widthMult);
            var fusedLayer = new FusedConvBNReLU(
                inChannels, oc, kernel,
                stride, padding,
                _quantizationBits);

            AddLayer(fusedLayer);
        }

        private void AddOptimizedInvertedResidual(int inChannels, int outChannels, int stride, float widthMult)
        {
            int ic = (int)(inChannels * widthMult);
            int oc = (int)(outChannels * widthMult);
            int expandRatio = 6;
            int hiddenDim = ic * expandRatio;

            var residualLayers = new ObservableCollection<Layer>();

            // Expansion phase
            if (expandRatio != 1)
            {
                if (_enableFusedOperations)
                {
                    residualLayers.Add(new FusedConvBNReLU(ic, hiddenDim, 1, 1, 0, _quantizationBits));
                }
                else
                {
                    AddConvBNReLU(ic, hiddenDim, 1, 1, 0, 1.0f, residualLayers);
                }
            }

            // Depthwise
            if (_enableFusedOperations)
            {
                residualLayers.Add(new FusedConvBNReLU(hiddenDim, hiddenDim, 3, stride, 1, _quantizationBits));
            }
            else
            {
                AddConvBNReLU(hiddenDim, hiddenDim, 3, stride, 1, 1.0f, residualLayers);
            }

            // Projection
            if (_enableFusedOperations)
            {
                residualLayers.Add(new FusedConvBNReLU(hiddenDim, oc, 1, 1, 0, _quantizationBits));
            }
            else
            {
                AddConvBNReLU(hiddenDim, oc, 1, 1, 0, 1.0f, residualLayers);
            }

            // Skip connection if dimensions match
            if (ic == oc && stride == 1)
            {
                AddLayer(new InvertedResidualBlock(residualLayers, ic));
            }
            else
            {
                foreach (var layer in residualLayers)
                {
                    AddLayer(layer);
                }
            }
        }

        private void AddConvBNReLU(int inChannels, int outChannels, int kernel, int stride, int padding,
        float widthMult, ObservableCollection<Layer> targetList = null)
        {
            int oc = (int)(outChannels * widthMult);

            var conv = new Conv2dMobileNet(oc, kernel, Dtype.float32, stride: stride,
                pad: padding, in_channels: inChannels);

            var shape = new Shape(oc);
            var batchNorm = new DeZero.NET.Layers.Normalization.BatchNorm(oc);
            batchNorm.InitParams(
                new Variable(xp.zeros(shape, dtype: Dtype.float32)),
                new Variable(xp.ones(shape, dtype: Dtype.float32)),
                new Variable(xp.ones(shape, dtype: Dtype.float32)),
                new Variable(xp.zeros(shape, dtype: Dtype.float32))
            );

            var relu = new DeZero.NET.Layers.Activation.ReLU6();

            if (targetList != null)
            {
                targetList.Add(conv);
                targetList.Add(batchNorm);
                targetList.Add(relu);
            }
            else
            {
                AddLayer(conv);
                AddLayer(batchNorm);
                AddLayer(relu);
            }
        }

        private void AddLayer(Layer layer)
        {
            _layers.Value.Add(layer);
            SetAttribute($"Layer_{_layerIndex++}", layer);
        }

        public override Variable[] Forward(params Variable[] inputs)
        {
            using var scope = new BatchProcessingScope(_batchSize);
            using var localScope = new ComputationScope();
            try
            {
                var x = inputs[0];
                x = Quantize(x);

                if (x.Shape[0] > _batchSize)
                {
                    localScope.Register(x);
                    return ProcessLargeBatch(x);
                }

                foreach (var layer in Layers)
                {
                    localScope.Register(x);
                    x = ProcessLayerWithOptimization(layer, x, scope);
                }

                _memoryManager.CheckAndCleanMemory();

                return new[] { x };
            }
            catch (Exception ex)
            {
                _logger.LogError($"Forward pass error: {ex.Message}");
                throw;
            }
        }

        private Variable ProcessLayerWithOptimization(Layer layer, Variable input, BatchProcessingScope scope)
        {
            using var layerScope = new ComputationScope();
            using var output = layer.Forward(input)[0];

            // 中間結果をすぐに解放
            var result = output.copy();

            if (layer is not FusedConvBNReLU)
            {
                layerScope.Register(result);
                result = Quantize(result);
            }

            return result;
        }

        private Variable[] ProcessLargeBatch(Variable x)
        {
            using var scope = new BatchProcessingScope(_batchSize);
            var results = new List<Variable>();
            int batchSize = x.Shape[0];

            try
            {
                for (int i = 0; i < batchSize; i += _batchSize)
                {
                    using var batchScope = new ComputationScope();
                    int end = Math.Min(i + _batchSize, batchSize);
                    using var batch = x.Data.Value.Slice(new[] { new Slice(i, end) }).ToVariable();
                    var result = ProcessSingleBatch(batch, scope);
                    results.Add(result);

                    if (i % (_batchSize * 4) == 0)
                    {
                        _memoryManager.ForceCleanup();
                    }
                }

                return new[] { ConcatenateResults(results) };
            }
            finally
            {
                foreach (var result in results)
                {
                    result?.Dispose();
                }
            }
        }

        private Variable ProcessSingleBatch(Variable batch, BatchProcessingScope parentScope)
        {
            using var scope = new ComputationScope();
            var x = batch;

            foreach (var layer in Layers)
            {
                scope.Register(x);
                x = ProcessLayerWithOptimization(layer, x, parentScope);
            }
            scope.Register(x);
            return x.copy();
        }

        private Variable ConcatenateResults(List<Variable> results)
        {
            try
            {
                var arrays = results.Select(r => r.Data.Value).ToArray();
                var concatenated = xp.concatenate(arrays, axis: 0);
                return concatenated.ToVariable();
            }
            finally
            {
                foreach (var result in results)
                {
                    result?.Dispose();
                }
            }
        }

        private Variable Quantize(Variable x)
        {
            using var scope = new ComputationScope();
            var data = x.Data.Value;
            using var min = data.min();
            using var max = data.max();

            var scale = (float)Math.Pow(2, _quantizationBits) - 1;
            using var eps = new NDarray(float.Epsilon);
            using var normalized = (data - min) / (max - min + eps);
            using var normalized_scale = xp.round(normalized * scale);
            using var quantized = normalized_scale / scale;
            var rescaled = quantized * (max - min + eps) + min;

            return new Variable(rescaled);
        }

        public void Dispose()
        {
            if (_disposed) return;
            _memoryManager.Dispose();
            _disposed = true;
            GC.SuppressFinalize(this);
        }
    }

    public class MemoryManager : IDisposable
    {
        private readonly ILogger _logger;
        private readonly int _threshold;
        private bool _disposed;

        public MemoryManager(ILogger logger, int threshold)
        {
            _logger = logger;
            _threshold = threshold;
        }

        public void CheckMemoryUsage()
        {
            var usage = GpuMemoryMonitor.Instance.GetCurrentMemoryUsage();
            if (usage > _threshold)
            {
                ForceCleanup();
            }
        }

        public void CheckAndCleanMemory()
        {
            CheckMemoryUsage();
            GC.Collect(0, GCCollectionMode.Optimized);
        }

        public void ForceCleanup()
        {
            GpuMemoryMonitor.ForceMemoryPool();
            GC.Collect(2, GCCollectionMode.Forced);
            Finalizer.Instance.Collect();
        }

        public void Dispose()
        {
            if (_disposed) return;
            ForceCleanup();
            _disposed = true;
        }
    }
}