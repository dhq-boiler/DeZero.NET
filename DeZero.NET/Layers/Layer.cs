using DeZero.NET.Core;
using System.Diagnostics;

namespace DeZero.NET.Layers
{
    public abstract class Layer
    {
        private List<WeakReference> inputs = new List<WeakReference>();
        private List<WeakReference> outputs = new List<WeakReference>();

        private HashSet<string> _params = new();
        private Dictionary<string, object> _dictionary = new();

        public virtual Func<Variable[], Variable[]> F => xs => Call(xs);

        public void SetAttribute(string name, object value, int depth = 0)
        {
            if (value is IWrap wrap)
            {
                SetAttribute($"{name}_{depth + 1}", wrap.Layer.Value, depth + 1);
            }
            if (value is Variable || value is Parameter || value is Layer)
            {
                _params.Add(name);
            }
            _dictionary[name] = value;
        }

        protected void RegisterEvent(params Property[] properties)
        {
            foreach (var notifyPropertyChanged in properties)
            {
                notifyPropertyChanged.ValueChanged += NotifyPropertyChangedOnValueChanged;
            }
        }

        private void NotifyPropertyChangedOnValueChanged(object sender, PropertyValueChangedEventArgs e)
        {
            SetAttribute(e.PropertyName, e.Value);
        }

        public abstract Variable[] Forward(params Variable[] xs);

        public Variable[] Call(params Variable[] inputs)
        {
            var outputs = Forward(inputs);
            // Updating references with weak references
            this.inputs = inputs.Select(input => new WeakReference(input)).ToList();
            this.outputs = outputs.ToList().Select(output => new WeakReference(output)).ToList();
            return outputs;
        }

        public IEnumerable<Parameter> Params()
        {
            foreach (var name in _params)
            {
                var obj = _dictionary[name];
                if (obj is Layer layer)
                {
                    foreach (var param in layer.Params())
                    {
                        yield return param;
                    }
                }
                else if (obj is Parameter param)
                {
                    yield return param;
                }
                else
                {
                    Debug.WriteLine($"{name} is not found.");
                }
            }
        }

        public void ClearGrads()
        {
            foreach (var param in Params())
            {
                param.ClearGrad();
            }
        }

        public void ToCpu()
        {
            foreach (var param in Params())
            {
                param.ToCpu();
            }
        }

        public void ToGpu()
        {
            foreach (var param in Params())
            {
                param.ToGpu();
            }
        }

        private void FlattenParams(Dictionary<string, Parameter> paramsDict, string parentKey = "")
        {
            foreach (var name in _params)
            {
                //var obj = GetType().GetProperty(name)?.GetValue(this);
                var obj = _dictionary[name];
                var key = string.IsNullOrEmpty(parentKey) ? name : $"{parentKey}/{name}";
                if (obj is Layer layer)
                {
                    layer.FlattenParams(paramsDict, key);
                }
                else if (obj is Parameter param)
                {
                    paramsDict[key] = param;
                }
            }
        }

        public void SaveWeights()
        {
            ToCpu();
            var paramsDict = new Dictionary<string, Parameter>();
            FlattenParams(paramsDict);
            var arrayDict = paramsDict.Where(pair => pair.Value != null)
                .ToDictionary(pair => pair.Key, pair => pair.Value.Data.Value);

            Directory.CreateDirectory("weights");

            foreach (var key in arrayDict.Keys)
            {
                var filename = Path.Combine("weights", Uri.EscapeDataString($"{key}.npy")).Replace("%2F", "_");
                Console.Write($"\n {filename} ...");
                var ndarray = arrayDict[key];
                xp.save(filename, ndarray);
                Console.Write("Done.");
            }
            Console.Write(Environment.NewLine);

            //// JSONファイルとしてシリアライズ
            //var options = new JsonSerializerOptions
            //{
            //    WriteIndented = false // 読みやすい形式で出力
            //};
            //var jsonString = JsonSerializer.Serialize(arrayDict, options);
            //File.WriteAllText(path, jsonString);
        }

        public void LoadWeights()
        {
            //var jsonString = File.ReadAllText(path);
            //var options = new JsonSerializerOptions
            //{
            //    PropertyNameCaseInsensitive = true // プロパティ名の大文字小文字を無視
            //};
            //var arrayDict = JsonSerializer.Deserialize<Dictionary<string, NDarrayDTO>>(jsonString, options);
            var paramsDict = new Dictionary<string, Parameter>();
            FlattenParams(paramsDict);
            foreach (var key in paramsDict.Keys)
            {
                var filename = Path.Combine("weights", Uri.EscapeDataString($"{key}.npy")).Replace("%2F", "_");
                Console.Write($"\n {filename} ...");
                paramsDict[key].Data.Value = xp.load(filename);
                Console.Write("Done.");
                //if (arrayDict.ContainsKey(key))
                //{
                //    paramsDict[key].Data.Value = arrayDict[key].ToNDarray();
                //}
            }
            Console.Write(Environment.NewLine);
        }

        public void DisposeAllInputs()
        {
            foreach (var input in inputs)
            {
                if (input.Target is Variable variable)
                {
                    variable.Dispose();
                }
            }
        }
    }
}
