using Cupy;
using DeZero.NET.Core;
using Numpy;
using Python.Runtime;
using System.Diagnostics;

namespace DeZero.NET
{
    public static class Utils
    {
        public static Variable as_variable(object obj)
        {
            if (obj is Variable v)
            {
                return v;
            }

            return new Variable((NDarray)obj);
        }

        public static NDarray sum_to(NDarray x, Shape shape)
        {
            int ndim = shape.Dimensions.Length;
            int lead = x.ndim - ndim;
            int[] leadAxis = Enumerable.Range(0, lead).ToArray();

            int[] axis = Enumerable.Range(0, shape.Dimensions.Length)
                .Where(i => shape[i] == 1)
                .Select(i => i + lead)
                .ToArray();

            // leadAxis と axis を結合
            int[] combinedAxes = leadAxis.Concat(axis).ToArray();

            NDarray y = x.sum(new Axis(combinedAxes), keepdims: true);
            if (lead > 0)
            {
                y = y.squeeze(new Axis(leadAxis));
            }
            return y;
        }

        //public static bool gradient_check(Function f, Variable x, Params kwargs, double rtol = 1e-4, double atol = 1e-5)
        //{
        //    return gradient_check(Params<Function, Variable, double, double>.args(f, x, rtol, atol).SetParams(kwargs));
        //}

        public static bool gradient_check(Function f, Params<Variable> _x, Params kwargs, double rtol = 1e-4, double atol = 1e-5)
        {
            //Function f = args.Get<Function>("f");
            //Variable x = args.Get<Variable>(1);
            //double rtol = args.Get<double>("rtol", 1e-4);
            //double atol = args.Get<double>("atol", 1e-5);
            //Params kwargs = args.Get<Params>("kwargs");
            Variable x = _x.Get<Variable>(0);

            x.Data = x.Data.astype(xp.float64);

            var num_grad = numerical_grad(f, Params<Variable>.args(x), kwargs);
            var y = f.Call(Params<Variable, Params>.args(x, kwargs));
            y[0].Backward();
            var bp_grad = x.Grad.Data;

            Debug.Assert(bp_grad.shape == num_grad.shape);
            var res = array_allclose(num_grad, bp_grad, atol: atol, rtol: rtol);

            if (!res)
            {
                Console.Error.WriteLine();
                Console.Error.WriteLine("========== FAILED (Gradient Check) ==========");
                Console.Error.WriteLine("Numerical Grad");
                Console.Error.WriteLine($" shape: {num_grad.shape.shape}");
                Console.Error.WriteLine($" values: {xp.take([num_grad.Switch().flatten()], [xp.array([1, 2, 3, 4, 5, 6, 7, 8])])} ...");
                Console.Error.WriteLine("Backprop Grad");
                Console.Error.WriteLine($" shape: {bp_grad.shape.shape}");
                Console.Error.WriteLine($" values: {xp.take([bp_grad.flatten()], [xp.array([1, 2, 3, 4, 5, 6, 7, 8])])} ...");
            }

            return res;
        }

        public static NDarray numerical_grad(Function f, Params<Variable> args, Params kwargs)
        {
            NDarray _x = args.Get<Variable>("x").Data;

            List<Variable> argsList = [..args.Through()];

            var eps = 1e-4;
            Numpy.NDarray np_x;
            if (Gpu.Available && Gpu.Use)
            {
                np_x = cpExtensions.asnumpy(_x.CupyNDarray);
            }
            else
            {
                np_x = _x.NumpyNDarray;
            }

            if (Gpu.Available && Gpu.Use)
            {
                Gpu.Use = false;
                argsList.ForEach(x => x.Data.Push(ArrayMode.np));
                Numpy.NDarray grad = Numpy.np.zeros_like(np_x);
                dynamic np = Py.Import("numpy");
                var flags = new PyList();
                flags.Append(new PyString("multi_index"));
                var op_flags = new PyList();
                op_flags.Append(new PyString("readwrite"));
                var it = np.nditer(np_x.data, flags: flags, op_flags: op_flags);
                while (!it.finished)
                {
                    int[] idx = IndexConverter.ConvertPyObjectToIntArray(it.multi_index);
                    var tmp_val = np_x[idx].copy();

                    np_x[idx] = tmp_val + eps;
                    var x = new Variable(new NDarray(np_x, false));
                    var y1 = f.Call(
                        Params<Variable>.args(x).SetParams<Params>(kwargs));
                    var y1arr = y1[0].Data.NumpyNDarray.copy();

                    np_x[idx] = tmp_val - eps; 
                    x = new Variable(new NDarray(np_x, false));
                    var y2 = f.Call(
                        Params<Variable>.args(x).SetParams<Params>(kwargs));
                    var y2arr = y2[0].Data.NumpyNDarray.copy();

                    var diff = (y1arr - y2arr).sum();
                    grad[idx] = diff / (2 * eps);

                    np_x[idx] = tmp_val;
                    it.iternext();
                }
                argsList.ForEach(x => x.Data.Pop());
                Gpu.Use = true;
                return new NDarray(grad);
            }
            else
            {
                argsList.ForEach(x => x.Data.Push(ArrayMode.np));
                Numpy.NDarray grad = Numpy.np.zeros_like(np_x);
                dynamic np = Py.Import("numpy");
                var flags = new PyList();
                flags.Append(new PyString("multi_index"));
                var op_flags = new PyList();
                op_flags.Append(new PyString("readwrite"));
                var it = np.nditer(np_x.data, flags: flags, op_flags: op_flags);
                while (!it.finished)
                {
                    int[] idx = IndexConverter.ConvertPyObjectToIntArray(it.multi_index);
                    var tmp_val = np_x[idx].copy();

                    np_x[idx] = tmp_val + eps;
                    var x = new Variable(new NDarray(np_x, false));
                    var y1 = f.Call(
                        Params<Variable>.args(x).SetParams<Params>(kwargs));
                    var y1arr = y1[0].Data.NumpyNDarray.copy();

                    np_x[idx] = tmp_val - eps;
                    x = new Variable(new NDarray(np_x, false));
                    var y2 = f.Call(
                        Params<Variable>.args(x).SetParams<Params>(kwargs));
                    var y2arr = y2[0].Data.NumpyNDarray.copy();

                    var diff = (y1arr - y2arr).sum();
                    grad[idx] = diff / (2 * eps);

                    np_x[idx] = tmp_val;
                    it.iternext();
                }
                argsList.ForEach(x => x.Data.Pop());
                return new NDarray(grad);
            }
        }

        private static PyList ToPyList(string[] multiIndex)
        {
            return new PyList(multiIndex.Select(x => new PyString(x)).ToArray());
        }

        private static PyList ToPyList(string[][] readwrite)
        {
            return new PyList(readwrite.Select(x =>
            {
                return new PyList(x.Select(y =>
                {
                    return new PyString(y);
                }).ToArray());
            }).ToArray());
        }

        public static bool array_allclose(Variable a, Variable b, double rtol = 1e-4, double atol = 1e-5)
        {
            return array_allclose(a.Data, b.Data, rtol, atol);
        }

        public static bool array_allclose(NDarray a, NDarray b, double rtol = 1e-4, double atol = 1e-5)
        {
            var (na, nb) = Gpu.Available && Gpu.Use ? (cpExtensions.asnumpy(a.CupyNDarray).copy(), b.ToNumpyNDarray.copy()) : (a.NumpyNDarray, b.NumpyNDarray);
            return np.allclose(na, nb, atol: (float)atol, rtol: (float)rtol);
        }

        public static Variable reshape_sum_backward(Variable gy, Shape x_shape, Axis axis, bool? keepdims)
        {
            var ndim = x_shape.Dimensions.Length;
            var tupled_axis = axis;
            if (axis is null)
            {
                tupled_axis = null;
            }

            List<int> shape; 

            if (!(ndim == 0 || tupled_axis is null || keepdims.HasValue && keepdims.Value))
            {
                var actual_axis = tupled_axis.Axes.Select(a => a >= 0 ? a : a + ndim);
                shape = gy.Shape.Dimensions.ToList();
                foreach (var a in actual_axis.OrderBy(b => b))
                {
                    shape.Insert(1, a);
                }
            }
            else
            {
                shape = gy.Shape.Dimensions.ToList();
            }

            gy = gy.reshape(shape.Select(x => new Shape(x)).ToArray())[0];

            return gy;
        }
    }
}
