using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Text;
using Cupy;
using DeZero.NET.Core;
using DeZero.NET.Functions;
using Python.Runtime;

namespace DeZero.NET
{
    public class Variable : PythonObject, INotifyPropertyChanged
    {
        public string Title { get; } = new Func<string>(() =>
        {
            string hiragana = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん";
            Random random = new Random();
            StringBuilder sb = new StringBuilder();

            for (int i = 0; i < 3; i++)
            {
                int index = random.Next(hiragana.Length);
                sb.Append(hiragana[index]);
            }

            return sb.ToString();
        })();

        private Function _Creator;
        public Property<NDarray> Data { get; } = new();
        public Property<string> Name { get; } = new();
        public Property<Variable> Grad { get; } = new();

        public Function Creator
        {
            get => _Creator;
            set
            {
                _Creator = value;
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
            Grad.Value = null;
        }

        public void Backward(bool retain_grad = false, bool create_graph = false)
        {
            if (Grad.Value is null)
            {
                Grad.Value = new Variable(xp.ones_like(Data.Value));
            }

            List<Function> funcs = [];
            HashSet<Function> seen_set = new();

            AddFunc(funcs, seen_set, Creator);
            while (funcs.Any())
            {
                var f = funcs.First();
                funcs.RemoveAt(0);
                var gys = f.Outputs.Select(o => o.Grad.Value).ToArray();

                using (var usingConfig = new UsingConfig("EnableBackprop", create_graph))
                {
                    var gxs = f.Backward(Params.New.SetPositionalArgs(gys));

                    foreach (var (x, gx) in f.Inputs.Select(p => p.Variable).Zip(gxs))
                    {
                        if (x is null)
                            continue;

                        if (x.Grad.Value is null)
                        {
                            x.Grad.Value = gx;
                        }
                        else
                        {
                            x.Grad.Value = x.Grad.Value + gx;
                        }

                        if (x.Creator is not null)
                        {
                            AddFunc(funcs, seen_set, x.Creator);
                        }
                    }
                }

                if (!retain_grad)
                {
                    foreach (var y in f.Outputs)
                    {
                        y.Grad.Value = null;
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
            this.Data.Value.NumpyNDarray = cpExtensions.asnumpy(this.Data.Value.CupyNDarray);
        }

        public void ToGpu()
        {
            this.Data.Value.CupyNDarray = this.Data.Value.NumpyNDarray.asarray();
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
    }
}
