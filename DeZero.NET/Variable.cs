using Cupy;
using DeZero.NET.Core;
using DeZero.NET.Extensions;
using DeZero.NET.Functions;
using Python.Runtime;
using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace DeZero.NET
{
    public class Variable : INotifyPropertyChanged, IDisposable, IDeZeroObject
    {
        private static readonly Random _random = new Random(Seed: 0);
        private Function _Creator;

        public int Title { get; set; } = _random.Next();
        public Property<string> Name { get; } = new(nameof(Name));
        public Property<NDarray> Data { get; } = new(nameof(Data));
        public Property<Variable> Grad { get; } = new(nameof(Grad));

        public Function Creator
        {
            get => _Creator;
            set
            {
                _Creator = value;
                if (value is null)
                {
                    return;
                }
                Generation = value.Generation + 1;
                _Creator._ForwardedTicks = DateTime.Now.Ticks;
            }
        }

        public int Generation { get; set; } = 0;

        public Variable(NDarray data, string name = null)
        {
            Data.Value = data;
            Name.Value = name;
        }

        public Shape Shape => Data.Value.shape;

        public int ndim => Data.Value.ndim;

        public int size => Data.Value.size;

        public Dtype Dtype => Data.Value.dtype;

        public int __len__ => Data.Value.len;

        public string __repr__
        {
            get
            {
                if (Data.Value is null)
                    return "variable(null)";
                return $"variable({Data.Value.ToString().Replace("\n", "\n         ")})";
            }
        }

        public void Unchain()
        {
            Creator = null;
        }

        public void ClearGrad()
        {
            Grad.Value?.Dispose();
            Grad.Value = null;
        }

        public void Backward(bool retain_grad = false, bool create_graph = false)
        {
            try
            {
                // 1. 自分の勾配が未設定の場合、全て1の勾配を作成
                if (Grad.Value is null)
                {
                    Grad.Value = new Variable(xp.ones_like(Data.Value));
                }

                // 2. 逆伝播の準備
                var graph = new BackwardComputationGraph();
                graph.AddFunction(Creator);

                // 3. 逆伝播の実行
                while (graph.HasNext())
                {
                    using var functionScope = new ComputationScope();
                    var function = graph.GetNext();

                    // 出力側の勾配を収集
                    var outputGradients = function.Outputs
                        .Select(output => output.Grad?.Value)
                        .ToArray();

                    // 勾配が全てnullの場合はスキップ
                    if (outputGradients.All(g => g is null || g.Data.Value is null))
                        continue;

                    // 逆伝播の実行
                    Variable[] inputGradients;
                    using (var config = new UsingConfig("EnableBackprop", create_graph))
                    {
                        inputGradients = function.Backward(Params.New.SetPositionalArgs(outputGradients));
                        functionScope.Register(inputGradients[0]);
                    }

                    // 入力側の勾配を更新
                    for (int i = 0; i < function.Inputs.Count(); i++)
                    {
                        var input = function.Inputs.ElementAt(i).Variable;
                        if (input is null) continue;

                        var gx = inputGradients[i];
                        if (gx?.Data?.Value is null) continue;

                        // 勾配のコピーを作成
                        using var newGrad = gx.Data.Value.copy().ToVariable();

                        if (input.Grad.Value is null)
                        {
                            input.Grad.Value = newGrad;
                        }
                        else
                        {
                            using var oldGrad = input.Grad.Value;
                            input.Grad.Value = oldGrad + newGrad;
                        }

                        // 次の関数を追加
                        if (input.Creator is not null)
                        {
                            graph.AddFunction(input.Creator);
                        }
                    }

                    // 不要な出力勾配の解放
                    if (!retain_grad && !function.Outputs.Any(o => graph.ShouldPreserve(o)))
                    {
                        foreach (var output in function.Outputs)
                        {
                            output.Grad.Value?.Dispose();
                            output.Grad.Value = null;
                        }
                    }

                    GC.Collect();
                    GpuMemoryMonitor.ForceMemoryPool();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Backward error: {ex.Message}");
                throw;
            }
        }

        private void InitializeGradientIfNull()
        {
            if (Grad.Value is null)
            {
                Grad.Value = new Variable(xp.ones_like(Data.Value));
            }
        }

        private class BackwardComputationGraph
        {
            private readonly Queue<Function> _functionQueue = new();
            private readonly HashSet<Function> _visitedFunctions = new();
            private readonly HashSet<Variable> _preservedVariables = new();

            public void AddFunction(Function function)
            {
                if (function is null || _visitedFunctions.Contains(function))
                    return;

                _functionQueue.Enqueue(function);
                _visitedFunctions.Add(function);

                // モデルのパラメータを保持対象として記録
                foreach (var input in function.Inputs)
                {
                    if (input.Variable?.Creator is null) // パラメータは Creator が null
                    {
                        _preservedVariables.Add(input.Variable);
                    }
                }
            }

            public bool HasNext() => _functionQueue.Count > 0;

            public Function GetNext() => _functionQueue.Dequeue();

            public bool ShouldPreserve(Variable variable) => _preservedVariables.Contains(variable);
        }

        //private void ProcessBackwardGraph(BackwardComputationGraph graph, bool retain_grad, bool create_graph)
        //{
        //    // 最初の関数（自分を作った関数）を追加
        //    graph.AddFunction(Creator);

        //    // 逆伝播を実行
        //    while (graph.FunctionQueue.Any())
        //    {
        //        var function = graph.FunctionQueue.First();
        //        graph.FunctionQueue.RemoveAt(0);

        //        // 1. 出力側の勾配を収集
        //        var outputGradients = function.Outputs
        //            .Select(output => output.Grad.Value)
        //            .ToArray();

        //        // 2. 逆伝播の実行（create_graphフラグに応じてモードを切り替え）
        //        Variable[] inputGradients;
        //        using (var config = new UsingConfig("EnableBackprop", create_graph))
        //        {
        //            inputGradients = function.Backward(Params.New.SetPositionalArgs(outputGradients));
        //        }

        //        // 3. 入力側の勾配を更新
        //        UpdateInputGradients(function, inputGradients, graph);

        //        // 4. 不要な勾配の解放（retain_gradがfalseの場合）
        //        if (!retain_grad)
        //        {
        //            ReleaseOutputGradients(function);
        //        }

        //        graph.DisposableFunctions.Add(function);
        //    }
        //}

        private void UpdateInputGradients(Function function, Variable[] gradients, BackwardComputationGraph graph)
        {
            foreach (var (input, gradient) in function.Inputs.Select(p => p.Variable).Zip(gradients))
            {
                if (input is null) continue;

                // 新しい勾配のコピーを作成
                if (gradient?.Data?.Value is not null)
                {
                    var newGradient = gradient.Data.Value.copy().ToVariable();

                    // 勾配を加算または設定
                    if (input.Grad.Value is null)
                    {
                        input.Grad.Value = newGradient;
                    }
                    else
                    {
                        input.Grad.Value += newGradient;
                    }
                }

                // 入力変数の作成関数があれば計算グラフに追加
                if (input.Creator is not null)
                {
                    graph.AddFunction(input.Creator);
                }
            }
        }

        private void ReleaseOutputGradients(Function function)
        {
            foreach (var output in function.Outputs)
            {
                output.Grad.Value?.Dispose();
                output.Grad.Value = null;
            }
        }

        public void UnchainBackward()
        {
            if (Creator is not null)
            {
                List<Function> funcs = [Creator];
                while (funcs.Any())
                {
                    var f = funcs.First();
                    funcs.RemoveAt(0);
                    foreach (var x in f.Inputs)
                    {
                        if (x.Variable.Creator is not null)
                        {
                            funcs.Append(x.Variable.Creator);
                            x.Variable.Unchain();
                        }
                    }
                }
            }
        }

        private void AddFunc(List<Function> funcs, HashSet<Function> seen_set, Function func)
        {
            if (func is null)
            {
                return;
            }

            if (!seen_set.Contains(func))
            {
                funcs.Add(func);
                seen_set.Add(func);

                var list = new List<Function>();
                list.AddRange(funcs.OrderBy(f => f.Generation).ThenByDescending(f => f._ForwardedTicks));

                funcs.Clear();
                funcs.AddRange(list);
            }
        }

        public Variable[] reshape(params Shape[] shapes)
        {
            if (Gpu.Available && Gpu.Use)
            {
                if (shapes.Length == 1 && (shapes[0].CupyShape is PyTuple || shapes[0].CupyShape is PyList))
                {
                    shapes = [shapes[0]];
                }

                return Reshape.Invoke(this, shapes.SelectMany(x => x.Dimensions).ToArray());
            }
            else
            {
                if (shapes.Length == 1 && (shapes[0].NumpyShape is PyTuple || shapes[0].NumpyShape is PyList))
                {
                    shapes = [shapes[0]];
                }

                return Reshape.Invoke(this, shapes.SelectMany(x => x.Dimensions).ToArray());
            }
        }

        public Variable[] transpose(params Axis[] axes)
        {
            if (Gpu.Available && Gpu.Use)
            {
                if (axes.Length == 0)
                {
                    axes = null;
                }
                else if (axes.Length == 1)
                {
                    if ((axes[0].CupyAxis is Tuple || axes[0].CupyAxis is PyList) || axes[0] is null)
                    {
                        axes = [axes[0]];
                    }
                }

                return Transpose.Invoke(this, axes);
            }
            else
            {
                if (axes.Length == 0)
                {
                    axes = null;
                }
                else if (axes.Length == 1)
                {
                    if ((axes[0].NumpyAxis is Tuple || axes[0].NumpyAxis is PyList) || axes[0] is null)
                    {
                        axes = [axes[0]];
                    }
                }

                return Transpose.Invoke(this, axes);
            }
        }



        public void ToCpu()
        {
            this.Data.Value.NumpyNDarray = this.Data.Value.ToNumpyNDarray;
        }

        public void ToGpu()
        {
            this.Data.Value.CupyNDarray = this.Data.Value.ToCupyNDarray;
        }

        public Variable T => new Variable(xp.transpose(Data.Value));

        public Variable pow(double power)
        {
            return Pow.Invoke(Data.Value.ToVariable(), new Variable(xp.array(power)))[0];
        }

        public Variable this[int index] => Gpu.Available && Gpu.Use ? new Variable(new NDarray(this.Data.Value.CupyNDarray[index])) : new Variable(new NDarray((this.Data.Value.NumpyNDarray)));

        public Variable this[(int x, int y) index] => Gpu.Available && Gpu.Use ? new Variable(new NDarray(this.Data.Value.CupyNDarray[index])) : new Variable(new NDarray((this.Data.Value.NumpyNDarray)));

        public override string ToString()
        {
            return __repr__;
        }

        public static Variable operator +(Variable a, Variable b)
        {
            return Add.Invoke(a, b).Item1[0];
        }

        public static Variable operator +(NDarray a, Variable b)
        {
            return Add.Invoke(a.ToVariable(), b).Item1[0];
        }

        public static Variable operator +(Variable a, NDarray b)
        {
            return Add.Invoke(a, b.ToVariable()).Item1[0];
        }

        public static Variable operator +(Variable a, double b)
        {
            return Add.Invoke(a, new Variable(new NDarray(b))).Item1[0];
        }

        public static Variable operator -(Variable a, Variable b)
        {
            return Sub.Invoke(a, b)[0];
        }

        public static Variable operator -(int a, Variable b)
        {
            return Sub.Invoke(xp.array(a).ToVariable(), b)[0];
        }

        public static Variable operator -(Variable a, NDarray b)
        {
            return Sub.Invoke(a, b.ToVariable())[0];
        }

        public static Variable operator -(NDarray a, Variable b)
        {
            return Sub.Invoke(a.ToVariable(), b)[0];
        }

        public static Variable operator *(Variable a, Variable b)
        {
            return Mul.Invoke(a, b)[0];
        }

        public static Variable operator *(NDarray a, Variable b)
        {
            return Mul.Invoke(a.ToVariable(), b)[0];
        }

        public static Variable operator *(Variable a, NDarray b)
        {
            return Mul.Invoke(a, b.ToVariable())[0];
        }

        public static Variable operator *(Variable a, double b)
        {
            return Mul.Invoke(a, new Variable(new NDarray(b)))[0];
        }

        public static Variable operator *(float a, Variable b)
        {
            return Mul.Invoke(new Variable(new NDarray(a)), b)[0];
        }

        public static Variable operator /(Variable a, Variable b)
        {
            return Div.Invoke(a, b)[0];
        }

        public static Variable operator /(NDarray a, Variable b)
        {
            return Div.Invoke(a.ToVariable(), b)[0];
        }

        public static Variable operator /(Variable a, NDarray b)
        {
            return Div.Invoke(a, b.ToVariable())[0];
        }
        public static Variable operator /(Variable a, double b)
        {
            return Div.Invoke(a, new Variable(new NDarray(b)))[0];
        }

        /// <summary>
        /// 単項演算子 - 
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public static Variable operator -(Variable x)
        {
            return Neg.Invoke(x)[0];
        }

        public override bool Equals(object obj)
        {
            if (obj is Variable v)
            {
                if (this.Data.Value is null || v.Data.Value is null)
                {
                    return false;
                }
                return this.Title.Equals(v.Title) && this.Data.Value.Equals(v.Data.Value);
            }
            else if (obj is NDarray arr)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return this.Data.Value.CupyNDarray.Equals(arr.CupyNDarray);
                }
                else
                {
                    return this.Data.Value.NumpyNDarray.Equals(arr.NumpyNDarray);
                }
            }
            return false;
        }

        public event PropertyChangedEventHandler? PropertyChanged;

        protected virtual void OnPropertyChanged([CallerMemberName] string? propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        protected bool SetField<T>(ref T field, T value, [CallerMemberName] string? propertyName = null)
        {
            if (EqualityComparer<T>.Default.Equals(field, value)) return false;
            field = value;
            OnPropertyChanged(propertyName);
            return true;
        }

        public new void Dispose()
        {
            GC.SuppressFinalize(this);
            Data?.Dispose();
            Grad?.Dispose();
        }
    }
}
