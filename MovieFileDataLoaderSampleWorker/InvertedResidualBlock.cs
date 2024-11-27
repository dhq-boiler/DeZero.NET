using DeZero.NET;
using DeZero.NET.Core;
using DeZero.NET.Layers;
using System.Collections.ObjectModel;

namespace MovieFileDataLoaderSampleWorker
{
    public class InvertedResidualBlock : Layer
    {
        public Property<ObservableCollection<Layer>> _layers { get; } = new(nameof(_layers));
        public Property<int> _in_channels { get; } = new(nameof(_in_channels));

        public InvertedResidualBlock(ObservableCollection<Layer> layers, int in_channels)
        {
            _layers.Value = new ObservableCollection<Layer>();
            _in_channels.Value = in_channels;

            foreach (var layer in layers)
            {
                _layers.Value.Add(layer);
                SetAttribute($"{layer.GetType().Name}", layer);
            }

            RegisterEvent(_layers, _in_channels);
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
