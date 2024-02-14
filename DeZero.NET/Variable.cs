

namespace DeZero.NET
{
    public class Variable : PythonObject
    {
        public NDarray Data { get; set; }
        public string Name { get; set; }
        public Variable Grad { get; set; }
        public Function Creator { get; set; }
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
                return $"variable({Data.str.Replace("\n", "\n         ")})";
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
                    var gxs = f.Backward(gys);

                    foreach (var (x, gx) in f.Inputs.Zip(gxs))
                    {
                        if (x.Grad is null)
                        {
                            x.Grad = gx;
                        }
                        else
                        {
                            x.Grad = x.Grad + gx;
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
                        if (x.Creator is not null)
                        {
                            funcs.Append(x.Creator);
                            x.Unchain();
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

        //Reshape

        //Transpose

        //T

        public static Variable operator +(Variable a, Variable b)
        {
            var c = a.Data + b.Data;
            return new Variable(c);
        }

        public static Variable operator -(Variable x)
        {
            return new Variable(xp.negative(x.Data));
        }
    }
}
