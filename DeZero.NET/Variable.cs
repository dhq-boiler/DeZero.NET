using Cupy;
using DeZero.NET.Core;
using DeZero.NET.Extensions;
using DeZero.NET.Functions;
using Python.Runtime;
using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using DocumentFormat.OpenXml.Presentation;

namespace DeZero.NET
{
    public class Variable : INotifyPropertyChanged, IDisposable, IDeZeroObject
    {
        private static readonly Random _random = new Random(Seed: 0);
        private Function _Creator;

        public int Title { get; set; } = _random.Next();
        public Property<string> Name { get; } = new(nameof(Name));
        public Property<NDarray> Data { get; } = new(nameof(Data));
        public Property<Variable> Grad { get; } = new (nameof(Grad));
        public Function[] Origins { get; set; }
        public Action<Variable> CopyGradToCloneSource { get; set; }

        //public event Action<Variable> GradChanged;

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

        public string StackTrace { get; set; }

        public Variable(NDarray data, string name = null)
        {
            Data.Value = data;
            Name.Value = name;
            Grad.ValueChanged += (s, e) =>
            {
                CopyGradToCloneSource?.Invoke(e.Value as Variable);
            };
            StackTrace = Environment.StackTrace;
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
            if (Grad.Value is null || initializeGrad)
            {
                Grad.Value = new Variable(xp.ones_like(Data.Value));
            }

            // 計算グラフのノードを保持するためのセット
            var seen_set = new HashSet<Function>();

            //Console.WriteLine($"depth:0 Backward:{Creator.GetType().Name}");
            var gys = Grad.Value;

            try
            {
                var gxs = Creator.Backward(Params.New.SetPositionalArgs(gys));

                foreach (var (x, gx) in Creator.Inputs.Select(p => p.Variable).Zip(gxs))
                {
                    if (x is null)
                        continue;

                    var newgrad = gx?.Data?.Value?.copy()?.ToVariable();
                    if (newgrad is not null)
                    {
                        if ((x.Grad.Value?.Data?.Value?.Array is Cupy.NDarray cpArray && cpArray.Handle != nint.Zero)
                            || (x.Grad.Value?.Data?.Value?.Array is Numpy.NDarray npArray && npArray.Handle != nint.Zero))
                        {
                            // 勾配の加算
                            x.Grad.Value = (x.Grad.Value.Data.Value + newgrad.Data.Value).ToVariable();
                            if (x.Creator is not null)
                            {
                                x.Creator.Outputs.Where(y => Utils.array_allclose(y.Data.Value, x.Data.Value)).ToList()
                                    .ForEach(y => y.Grad.Value = x.Grad.Value);
                            }
                        }
                        else
                        {
                            x.Grad.Value = newgrad;
                            if (x.Creator is not null)
                            {
                                x.Creator.Outputs.Where(y => Utils.array_allclose(y.Data.Value, x.Data.Value)).ToList()
                                    .ForEach(y => y.Grad.Value = x.Grad.Value);
                            }
                        }

                        // メモリリーク防止のための後処理
                        if (!retain_grad)
                        {
                            newgrad = null;
                        }
                    }

                    if (x.Creator is not null && x.Creator.Outputs.Any(o => Utils.array_allclose(x, o)))
                    {
                        //Console.WriteLine("Branch");
                        BackwardInternal(x.Creator, seen_set, retain_grad, 1);
                    }
                    //else
                    //{
                    //    Console.WriteLine("Reaf");
                    //}
                }
            }
            catch (Exception ex)
            {
                throw new Exception($"Error in backward pass at {Creator.GetType().Name}", ex);
            }

            foreach (var origin in Origins.Where(x => x is not null))
            {
                foreach (var output in origin.Outputs)
                {
                    BackwardInternal(output.Creator, seen_set, retain_grad, 1);
                }
            }

            // retain_gradがfalseの場合、中間の勾配をクリア
            if (!retain_grad)
            {
                ClearGrads(new HashSet<Variable>());
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

        private void BackwardInternal(Function over, HashSet<Function> seen_set, bool retain_grad, int depth)
        {
            if (seen_set.Contains(over))
            {
                return; // 既に処理済みのノードはスキップ
            }
            seen_set.Add(over);

            //Console.WriteLine($"depth:{depth} Backward:{over.GetType().Name}");

            // 勾配の計算
            var gys = over.Outputs.Select(x => x.Grad.Value).ToArray();

            if (gys.Any(x => x is null))
            {
                return;
            }

            try
            {
                var gxs = over.Backward(Params.New.SetPositionalArgs(gys));

                foreach (var (input, grad) in over.Inputs.Zip(gxs))
                {
                    if (input.Variable.Shape != grad.Shape)
                    {
                        Debugger.Break();
                    }
                }

                foreach (var (x, gx) in over.Inputs.Select(p => p.Variable).Zip(gxs))
                {
                    if (x is null)
                        continue;

                    var newgrad = gx?.Data?.Value?.copy()?.ToVariable();
                    if (newgrad is not null)
                    {
                        if ((x.Grad.Value?.Data?.Value?.Array is Cupy.NDarray cpArray && cpArray.Handle != nint.Zero)
                            || (x.Grad.Value?.Data?.Value?.Array is Numpy.NDarray npArray && npArray.Handle != nint.Zero))
                        {
                            // 勾配の加算
                            x.Grad.Value = (x.Grad.Value.Data.Value + newgrad.Data.Value).ToVariable();
                            if (x.Creator is not null)
                            {
                                x.Creator.Outputs.Where(y => Utils.array_allclose(y.Data.Value, x.Data.Value)).ToList()
                                    .ForEach(y => y.Grad.Value = x.Grad.Value);
                            }
                        }
                        else
                        {
                            x.Grad.Value = newgrad;
                            if (x.Creator is not null)
                            {
                                x.Creator.Outputs.Where(y => Utils.array_allclose(y.Data.Value, x.Data.Value)).ToList()
                                    .ForEach(y => y.Grad.Value = x.Grad.Value);
                            }
                        }

                        // メモリリーク防止のための後処理
                        if (!retain_grad)
                        {
                            newgrad = null;
                        }
                    }

                    if (x.Creator is not null && x.Creator.Outputs.Any(o => Utils.array_allclose(x, o)))
                    {
                        //Console.WriteLine("Branch");
                        BackwardInternal(x.Creator, seen_set, retain_grad, depth + 1);
                    }
                    else
                    {
                        //Console.WriteLine("Reaf");
                    }
                }

                foreach (var origin in Origins.Where(x => x is not null))
                {
                    foreach (var output in origin.Outputs)
                    {
                        BackwardInternal(output.Creator, seen_set, retain_grad, 1);
                    }
                }
            }
            catch (Exception ex)
            {
                throw new Exception($"Error in backward pass at {over.GetType().Name}", ex);
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

        public Variable copy(bool holdReference = true)
        {
            var ret = new Variable(Data.Value.copy())
            {
                Name = {
                    Value = Name.Value,
                },
                Grad = {
                    //Value = Grad.Value is null || Grad.Value.Data.Value is null ? null : Grad.Value.copy(),
                    //Value = Grad.Value?.Data.Value is null ? null : Grad.Value.Data.Value.copy().ToVariable(),
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
                    //var thiz = CreatorList.LastOrDefault()?.Outputs?.FirstOrDefault(x => x.Data.Value.Equals(this.Data.Value));
                    //if (thiz is not null && thiz.Grad.Value is null)
                    //{
                    //    thiz.Grad.Value = newGrad;
                    //}
                    //foreach (var origin in Origins.Where(x => x is not null))
                    //{
                    //    origin.Outputs.ToList().ForEach(x => x.Grad.SetValueWithNoFireEvent(newGrad));
                    //}
                };
            }

            return ret;
        }
    }
}
