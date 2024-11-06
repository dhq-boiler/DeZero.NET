using DeZero.NET.Core;
using DeZero.NET.Models;
using DeZero.NET.Optimizers.Converters;
using DeZero.NET.Optimizers.HookFunctions;
using System.Text.Json;

namespace DeZero.NET.Optimizers
{
    public abstract class Optimizer
    {
        public List<Property> SerializableParameters { get; } = new();
        public Model Target { get; set; }
        public List<HookFunction> Hooks { get; set; }

        protected Optimizer()
        {
            this.Target = null;
            this.Hooks = new List<HookFunction>();
        }

        public Optimizer Setup(Model target)
        {
            this.Target = target;
            return this;
        }

        public virtual void Update(Params args)
        {
            var _params = args?.Through?.Select(p => new DeZero.NET.Parameter(p.Variable))?.ToArray() ?? Target.Params().Where(p => p.Grad.Value is not null).ToArray();

            foreach (var f in Hooks)
            {
                f.Call(_params);
            }

            foreach (var param in _params)
            {
                UpdateOne(param);
            }
        }

        public abstract void UpdateOne(Parameter param);

        public void AddHook(HookFunction f)
        {
            Hooks.Add(f);
        }

        protected void RegisterNonVolatileParameters(params Property[] parameters)
        {
            SerializableParameters.AddRange(parameters);
        }

        public virtual void SaveParameters()
        {
            Directory.CreateDirectory("optimizer");

            foreach (var parameter in SerializableParameters)
            {
                
                if (parameter.Value is Dictionary<string, Variable> dic)
                {
                    var options = new JsonSerializerOptions();
                    options.Converters.Add(new KeyValuePairJsonConverter());

                    var keyValuePairs = dic.ToList();
                    //dicをJsonに変換して保存
                    var json = JsonSerializer.Serialize(keyValuePairs, options);
                    File.WriteAllText(Path.Combine("optimizer", $"{parameter.PropertyName}.json"), json);


                    foreach (var (key, value) in dic)
                    {
                        var filename = Path.Combine("optimizer", Uri.EscapeDataString($"{parameter.PropertyName}__{key}.npy")).Replace("%2F", "_");
                        Console.Write($"\n {filename} ...");
                        var ndarray = value.Data.Value;
                        xp.save(filename, ndarray);
                        Console.WriteLine("Done.");
                    }
                }
            }
        }

        public virtual void LoadParameters()
        {
            foreach (var parameter in SerializableParameters)
            {
                if (parameter.Value is Dictionary<string, Variable> dic)
                {
                    var options = new JsonSerializerOptions();
                    options.Converters.Add(new KeyValuePairJsonConverter());

                    //jsonを読み込んでDictionaryに変換
                    var json = File.ReadAllText(Path.Combine("optimizer", $"{parameter.PropertyName}.json"));
                    var otherDic = JsonSerializer.Deserialize<List<KeyValuePair<string, Variable>>>(json, options);
                    
                    //ダミーのVariableをディクショナリに追加する
                    otherDic.ToList().ForEach(pair => dic[pair.Key] = new Variable(xp.array(0)));

                    foreach (var (key, value) in dic)
                    {
                        try
                        {
                            var filename = Path.Combine("optimizer", Uri.EscapeDataString($"{parameter.PropertyName}__{key}.npy")).Replace("%2F", "_");
                            Console.Write($"\n {filename} ...");
                            var ndarray = xp.load(filename, allow_pickle: true);
                            value.Data.Value = ndarray;
                            Console.WriteLine("Done.");
                        }
                        catch (Exception e)
                        {
                            Console.WriteLine("Error while loading parameters.");
                            Console.WriteLine(e.Message);
                        }
                    }
                }
            }
        }
    }
}
