using DeZero.NET.Core;
using DeZero.NET.Extensions;
using DeZero.NET.Functions;
using Python.Runtime;
using System.Collections.Concurrent;
using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;
using DeZero.NET.Log;

namespace DeZero.NET
{
    public class Variable : INotifyPropertyChanged, IDisposable, IDeZeroObject
    {
        private bool _disposed;
        private readonly object _disposeLock = new object();
#if DEBUG
        private static readonly ConcurrentQueue<(Variable, string)> _cleanupQueue = new();
#else
        private static readonly ConcurrentQueue<Variable> _cleanupQueue = new();
#endif
        private static readonly Thread _cleanupThread;

        static Variable()
        {
            // クリーンアップスレッドの初期化
            _cleanupThread = new Thread(() =>
            {
                while (true)
                {
                    try
                    {
                        if (_cleanupQueue.IsEmpty)
                        {
                            Thread.Sleep(100);
                            continue;
                        }

                        using (Py.GIL())
                        {
                            while (_cleanupQueue.TryDequeue(out var variable))
                            {
                                try
                                {
#if DEBUG
                                    variable.Item1.ReleaseUnmanagedResources();
                                    variable.Item1.DisposedStackTrace = variable.Item2;
#else
                                    variable.ReleaseUnmanagedResources();
#endif
                                }
                                catch (Exception ex)
                                {
                                    Debug.WriteLine($"Error cleaning up Variable: {ex.Message}");
                                }
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        Debug.WriteLine($"Error in Variable cleanup thread: {ex.Message}");
                        Thread.Sleep(1000);
                    }
                }
            })
            {
                IsBackground = true,
                Name = "Variable Resource Cleanup"
            };
            _cleanupThread.Start();
        }

        private void ReleaseUnmanagedResources()
        {
            try
            {
                // Data propertyの解放
                if (Data?.Value is not null)
                {
#if DEBUG
                    PythonObjectTracker.UnTrackPythonObject(Data.Value);
                    InstanceTracker<NDarray>.Instance.Unregister(Data.Value);
#endif
                    Data.Value.Dispose();
                    Data.Value = null;
                }

                // Grad propertyの解放
                if (Grad?.Value != null)
                {
                    if (Grad.Value.Data?.Value is not null)
                    {
#if DEBUG
                        PythonObjectTracker.UnTrackPythonObject(Grad.Value.Data.Value);
                        InstanceTracker<NDarray>.Instance.Unregister(Grad.Value.Data.Value);
#endif
                        Grad.Value.Data.Value.Dispose();
                    }
                    Grad.Value.Dispose();
                    Grad.Value = null;
                }

                // 循環参照の解除
                Creator = null;
                CreatorList?.Clear();
                Origins = null;
                CopyGradToCloneSource = null;
#if DEBUG
                DisposedStackTrace = Environment.StackTrace;
#endif
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error releasing Variable resources: {ex.Message}");
            }
        }

        public void Dispose()
        {
            if (_disposed) return;

            lock (_disposeLock)
            {
                if (!_disposed)
                {
                    try
                    {
                        using (Py.GIL())
                        {
                            ReleaseUnmanagedResources();
                        }
                    }
                    catch (PythonException)
                    {
                        // GILが取得できない場合はクリーンアップキューに追加
#if DEBUG
                        _cleanupQueue.Enqueue((this, Environment.StackTrace));
#else
                        _cleanupQueue.Enqueue(this);
#endif
                    }
                    finally
                    {
                        _disposed = true;
                        GC.SuppressFinalize(this);
                    }
                }
            }
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    try
                    {
                        using (Py.GIL())
                        {
                            ReleaseUnmanagedResources();
                        }
                    }
                    catch (PythonException)
                    {
#if DEBUG
                        _cleanupQueue.Enqueue((this, Environment.StackTrace));
#else
                        _cleanupQueue.Enqueue(this);
#endif
                    }
                }

                _disposed = true;
            }
        }


        private static readonly Random _random = new Random(Seed: 0);
        private Function _Creator;

        public int Title { get; set; } = _random.Next();
        public Property<string> Name { get; } = new(nameof(Name));
        public Property<NDarray> Data { get; } = new(nameof(Data));
        public Property<Variable> Grad { get; } = new (nameof(Grad));
        public Function[] Origins { get; set; }
        public Action<Variable> CopyGradToCloneSource { get; set; }
        
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

        public List<Function> CreatorList { get; internal set; } = new();

        public int Generation { get; set; } = 0;

#if DEBUG
        public string StackTrace { get; set; }
        public string DisposedStackTrace { get; set; }
#endif

        public Variable(NDarray data, string name = null)
        {
            Data.Value = data;
            Name.Value = name;
            Grad.ValueChanged += (s, e) =>
            {
                CopyGradToCloneSource?.Invoke(e.Value as Variable);
            };
#if DEBUG
            StackTrace = Environment.StackTrace;
#endif
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

        public void Backward(bool retain_grad = false, bool create_graph = false, bool initializeGrad = false)
        {
            var logger = BackwardLogger.Instance;
            logger.Log("=== Starting Backward Pass ===");
            logger.Log($"[0] {Creator.GetType().Name} [Start]");

            if (Grad.Value is null || initializeGrad)
            {
                Grad.Value = new Variable(xp.ones_like(Data.Value));
            }

            var seen_set = new HashSet<Function>();
            var gys = Grad.Value;

            try
            {
                var gxs = Creator.Backward(Params.New.SetPositionalArgs(gys));

                for (int i = 0; i < Creator.Inputs.Count(); i++)
                {
                    var x = Creator.Inputs.ElementAt(i).Variable;
                    if (x is null) continue;

                    logger.Log($"[1] Branch {i + 1}: {x.GetType().Name}");

                    var newgrad = gxs.ElementAt(i)?.Data?.Value?.copy()?.ToVariable();
                    if (newgrad is not null)
                    {
                        if ((x.Grad.Value?.Data?.Value?.Array is Cupy.NDarray cpArray && cpArray.Handle != nint.Zero)
                            || (x.Grad.Value?.Data?.Value?.Array is Numpy.NDarray npArray && npArray.Handle != nint.Zero))
                        {
                            x.Grad.Value = (x.Grad.Value.Data.Value + newgrad.Data.Value).ToVariable();
                            if (x.Creator is not null)
                            {
                                x.Creator.Outputs.Where(y => y.Title == x.Title).ToList()
                                    .ForEach(y => y.Grad.Value = x.Grad.Value);
                            }
                        }
                        else
                        {
                            x.Grad.Value = newgrad;
                            if (x.Creator is not null)
                            {
                                x.Creator.Outputs.Where(y => y.Title == x.Title).ToList()
                                    .ForEach(y => y.Grad.Value = x.Grad.Value);
                            }
                        }

                        if (!retain_grad)
                        {
                            newgrad = null;
                        }
                    }

                    if (x.Creator is not null && x.Creator.Outputs.Any(o => x.Title == o.Title))
                    {
                        logger.Log($"[1] Continue: {x.Creator.GetType().Name}");
                        BackwardInternal(x.Creator, seen_set, retain_grad, 2);
                    }
                    else
                    {
                        logger.Log($"[1] Leaf: {x.Title}");
                    }
                }
            }
            catch (Exception ex)
            {
                logger.Log($"[ERROR] in {Creator.GetType().Name}: {ex.Message}");
                throw new Exception($"Error in backward pass at {Creator.GetType().Name}", ex);
            }

            foreach (var origin in Origins.Where(x => x is not null))
            {
                logger.Log($"[1] Origin: {origin.GetType().Name}");
                foreach (var output in origin.Outputs)
                {
                    BackwardInternal(output.Creator, seen_set, retain_grad, 2);
                }
            }

            if (!retain_grad)
            {
                logger.Log("[1] Clearing intermediate gradients");
                ClearGrads(new HashSet<Variable>());
            }

            logger.Log("=== Completed Backward Pass ===");
        }

        private void BackwardInternal(Function over, HashSet<Function> seen_set, bool retain_grad, int depth)
        {
            var logger = BackwardLogger.Instance;
            if (seen_set.Contains(over))
            {
                logger.Log($"[{depth}] Cycle: {over.GetType().Name}");
                return;
            }
            seen_set.Add(over);

            // 関数の種類を出力
            logger.Log($"[{depth}] Processing: {over.GetType().Name}");

            // 入力変数の情報を出力
            foreach (var input in over.Inputs)
            {
                if (input.Variable != null)
                {
                    var varInfo = new StringBuilder();
                    varInfo.AppendLine($"[{depth}] Input Details:");
                    varInfo.AppendLine($"  - ID: {input.Variable.Title}");
                    varInfo.AppendLine($"  - Name: {input.Name}");
                    varInfo.AppendLine($"  - Shape: {input.Variable.Shape}");
                    varInfo.AppendLine($"  - Type: {input.Variable.GetType().Name}");

                    // パラメータの場合は詳細情報を追加
                    if (input.Variable is Parameter p)
                    {
                        varInfo.AppendLine($"  - Parameter Name: {p.Name}");
                        //varInfo.AppendLine($"  - Layer: {p?.GetType().Name}");
                        varInfo.AppendLine($"  - Role: {(p.Name?.Value.Contains("b") ?? false ? "Bias" : "Weight")}");
                    }

                    // 勾配の状態
                    if (input.Variable.Grad.Value != null)
                    {
                        varInfo.AppendLine($"  - Gradient Shape: {input.Variable.Grad.Value.Shape}");
                        varInfo.AppendLine($"  - Gradient Stats: min={input.Variable.Grad.Value.Data.Value.min()}, max={input.Variable.Grad.Value.Data.Value.max()}");
                    }

                    logger.Log(varInfo.ToString());
                }
            }

            // 勾配の状態を出力
            var gys = over.Outputs.Select(x => x.Grad.Value).ToArray();
            logger.Log($"[{depth}] Gradient exists: {gys.All(x => x != null)}");

            if (gys.Any(x => x is null))
            {
                logger.Log($"[{depth}] Warning: Null gradient detected");
                return;
            }

            try
            {

                var gxs = over.Backward(Params.New.SetPositionalArgs(gys));

                foreach (var (input, grad) in over.Inputs.Zip(gxs))
                {
                    if (input.Variable.Shape != grad.Shape)
                    {
                        logger.Log($"[{depth}] Error: Shape mismatch {input.Variable.Shape} != {grad.Shape}");
                        Debugger.Break();
                    }
                }

                for (int i = 0; i < over.Inputs.Count(); i++)
                {
                    var x = over.Inputs.ElementAt(i).Variable;
                    if (x is null) continue;

                    var newgrad = gxs.ElementAt(i)?.Data?.Value?.copy()?.ToVariable();
                    logger.Log($"[{depth}] Computed gradient for {x.Title}: Shape={newgrad?.Shape}, IsNull={newgrad == null}");
                    if (newgrad is not null)
                    {
                        if ((x.Grad.Value?.Data?.Value?.Array is Cupy.NDarray cpArray && cpArray.Handle != nint.Zero)
                            || (x.Grad.Value?.Data?.Value?.Array is Numpy.NDarray npArray && npArray.Handle != nint.Zero))
                        {
                            x.Grad.Value = (x.Grad.Value.Data.Value + newgrad.Data.Value).ToVariable();
                        }
                        else
                        {
                            x.Grad.Value = newgrad;
                        }

                        if (x.Creator is not null)
                        {
                            x.Creator.Outputs.Where(y => y.Title == x.Title).ToList()
                                .ForEach(y => y.Grad.Value = x.Grad.Value);
                        }

                        if (!retain_grad)
                        {
                            newgrad = null;
                        }
                    }

                    if (x.Creator is not null && x.Creator.Outputs.Any(o => x.Title == o.Title))
                    {
                        logger.Log($"[{depth}] Continue: {x.Creator.GetType().Name}");
                        BackwardInternal(x.Creator, seen_set, retain_grad, depth + 1);
                    }
                    else
                    {
                        logger.Log($"[{depth}] Leaf: {x.Title}");
                    }
                }

                foreach (var origin in Origins.Where(x => x is not null))
                {
                    logger.Log($"[{depth}] Origin: {origin.GetType().Name}");
                    foreach (var output in origin.Outputs)
                    {
                        BackwardInternal(output.Creator, seen_set, retain_grad, depth + 1);
                    }
                }
            }
            catch (Exception ex)
            {
                logger.Log($"[{depth}] ERROR in {over.GetType().Name}: {ex.Message}");
                throw new Exception($"Error in backward pass at {over.GetType().Name}", ex);
            }
        }

        private void ClearGrads(HashSet<Variable> seen_vars)
        {
            // この変数が既に処理済みの場合はスキップ
            if (seen_vars.Contains(this))
            {
                return;
            }

            // この変数を処理済みとしてマーク
            seen_vars.Add(this);

            // この変数の勾配をクリア
            Grad.Value = null;

            // この変数が関数の出力である場合、その関数の入力変数も再帰的にクリア
            if (Creator is not null)
            {
                foreach (var input in Creator.Inputs)
                {
                    if (input.Variable is not null)
                    {
                        input.Variable.ClearGrads(seen_vars);
                    }
                }
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

        public Variable[] reshape(Shape shape)
        {
            if (Shape.Dimensions.SequenceEqual(shape.Dimensions))
            {
                return [this];
            }

            if (Gpu.Available && Gpu.Use)
            {
                return [new NDarray(this.Data.Value.ToCupyNDarray.reshape(shape.Dimensions.ToArray())).ToVariable()];
            }
            else
            {
                return [new NDarray(this.Data.Value.ToNumpyNDarray.reshape(shape.Dimensions.ToArray())).ToVariable()];
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

        public Variable copy(bool holdReference = true)
        {
            var ret = new Variable(Data.Value.copy())
            {
                Title = Title,
                Name = {
                    Value = Name.Value,
                },
                Grad = {
                    Value = Grad.Value is null ? null : Grad.Value
                },
                Creator = Creator,
                CreatorList = CreatorList,
                Generation = Generation,
                Origins = Origins
            };

            if (holdReference)
            {
                ret.CopyGradToCloneSource = newGrad =>
                {
                    Grad.SetValueWithNoFireEvent(newGrad);
                };
            }

            return ret;
        }
    }
}
