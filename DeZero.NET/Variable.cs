using Cupy;
using DeZero.NET.Core;
using DeZero.NET.Functions;
using Python.Runtime;

namespace DeZero.NET
{
    public class Variable : PythonObject
    {
        private Function _Creator;
        public NDarray Data { get; set; }
        public string Name { get; set; }
        public Variable Grad { get; set; }

        public Function Creator
        {
            get => _Creator;
            set
            {
                _Creator = value;
                Generation = value.Generation + 1;
            }
        }

        public int Generation { get; set; } = 0;

        public Variable(NDarray data, string name = null)
        {
            Data = data;
            Name = name;
        }

        public Shape Shape => Data.shape;

        public int ndim => Data.ndim;

        public int size => Data.size;

        public Dtype Dtype => Data.dtype;

        public int __len__ => Data.len;

        public string __repr__
        {
            get
            {
                if (Data is null)
                    return "variable(null)";
                return $"variable({Data.ToString().Replace("\n", "\n         ")})";
            }
        }

        public void Unchain()
        {
            Creator = null;
        }

        public void ClearGrad()
        {
            Grad = null;
        }

        public void Backward(bool retain_grad = false, bool create_graph = false)
        {
            if (Grad is null)
            {
                Grad = new Variable(xp.ones_like(Data));
            }

            List<Function> funcs = [];
            HashSet<Function> seen_set = new();

            AddFunc(funcs, seen_set, Creator);
            while (funcs.Any())
            {
                var f = funcs.First();
                funcs.RemoveAt(0);
                var gys = f.Outputs.Select(o => o.Grad).ToArray();

                using (var usingConfig = new UsingConfig("EnableBackprop", create_graph))
                {
                    var gxs = f.Backward(OrderedParams<Variable>.args(gys));

                    foreach (var (x, gx) in f.Inputs.Zip(gxs))
                    {
                        if (x.Variable.Grad is null)
                        {
                            x.Variable.Grad = gx;
                        }
                        else
                        {
                            x.Variable.Grad = x.Variable.Grad + gx;
                        }

                        if (x.Variable.Creator is not null)
                        {
                            AddFunc(funcs, seen_set, x.Variable.Creator);
                        }
                    }
                }

                if (!retain_grad)
                {
                    foreach (var y in f.Outputs)
                    {
                        y.Grad = null;
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
                list.AddRange(funcs.OrderBy(f => f.Generation));

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
            this.Data.NumpyNDarray = cpExtensions.asnumpy(this.Data.CupyNDarray);
        }

        public void ToGpu()
        {
            this.Data.CupyNDarray = this.Data.NumpyNDarray.asarray();
        }

        public Variable T => new Variable(xp.transpose(Data));

        public Variable pow(double power)
        {
            return Pow.Invoke(Data.ToVariable(), new Variable(xp.array(power)))[0];
        }

        public Variable this[int index] => Gpu.Available && Gpu.Use ? new Variable(new NDarray(this.Data.CupyNDarray[index])) : new Variable(new NDarray((this.Data.NumpyNDarray)));

        public Variable this[(int x, int y) index] => Gpu.Available && Gpu.Use ? new Variable(new NDarray(this.Data.CupyNDarray[index])) : new Variable(new NDarray((this.Data.NumpyNDarray)));

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
                return xp.equal(this.Data, v.Data).asscalar<bool>();
            }
            else if (obj is NDarray arr)
            {
                return xp.equal(this.Data, arr).asscalar<bool>();
            }
            return false;
        }

        //public static explicit operator Variable(NDarray arr)
        //{
        //    return arr.ToVariable();
        //}

        //public static explicit operator Variable(Core.Parameter p)
        //{
        //    return p.Variable;
        //}
    }
}
