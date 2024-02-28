﻿using Cupy;
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

        public static bool gradient_check(Function f, Variable x, double rtol = 1e-4, double atol = 1e-5, params Variable[] args)
        {
            x.Data = x.Data.astype(xp.float64);

            var num_grad = numerical_grad(f, x, args);
            var y = f.BaseForward([x, ..args]);
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

        public static NDarray numerical_grad(Function f, Variable x, params Variable[] args)
        {
            return numerical_grad(f, x.Data, args);
        }

        public static NDarray numerical_grad(Function f, NDarray x, params Variable[] args)
        {
            var eps = 1e-4;
            Numpy.NDarray np_x;
            if (Gpu.Available && Gpu.Use)
            {
                np_x = cpExtensions.asnumpy(x.CupyNDarray);
            }
            else
            {
                np_x = x.NumpyNDarray;
            }

            if (Gpu.Available && Gpu.Use)
            {
                Gpu.Use = false;
                args.ToList().ForEach(x => x.Data.Push(ArrayMode.np));
                Numpy.NDarray grad = Numpy.np.zeros_like(np_x);
                dynamic np = Py.Import("numpy");
                var flags = new PyList();
                flags.Append(new PyString("multi_index"));
                var op_flags = new PyList();
                op_flags.Append(new PyString("readwrite"));
                var it = np.nditer(np_x.data, flags: flags, op_flags: op_flags);
                while (!it.finished)
                {
                    var idx = it.multi_index;
                    var tuple2 = (Tuple<int, int>)NDarray.ToCsharp<Tuple<int, int>>(idx);
                    var tmp_val = np_x[tuple2.Item1, tuple2.Item2].copy();

                    np_x[tuple2.Item1, tuple2.Item2] = tmp_val + eps;
                    var y1 = f.BaseForward([new Variable(new NDarray(np_x, false)), .. args]);
                    var y1arr = y1[0].Data.NumpyNDarray.copy();

                    np_x[tuple2.Item1, tuple2.Item2] = tmp_val - eps;
                    var y2 = f.BaseForward([new Variable(new NDarray(np_x, false)), .. args]);
                    var y2arr = y2[0].Data.NumpyNDarray.copy();

                    var diff = (y1arr - y2arr).sum();
                    grad[tuple2.Item1, tuple2.Item2] = diff / (2 * eps);

                    np_x[tuple2.Item1, tuple2.Item2] = tmp_val;
                    it.iternext();
                }
                args.ToList().ForEach(x => x.Data.Pop());
                Gpu.Use = true;
                return new NDarray(grad);
            }
            else
            {
                args.ToList().ForEach(x => x.Data.Push(ArrayMode.np));
                Numpy.NDarray grad = Numpy.np.zeros_like(np_x);
                dynamic np = Py.Import("numpy");
                var flags = new PyList();
                flags.Append(new PyString("multi_index"));
                var op_flags = new PyList();
                op_flags.Append(new PyString("readwrite"));
                var it = np.nditer(np_x.data, flags: flags, op_flags: op_flags);
                while (!it.finished)
                {
                    var idx = it.multi_index;
                    var tuple2 = (Tuple<int, int>)NDarray.ToCsharp<Tuple<int, int>>(idx);
                    var tmp_val = np_x[tuple2.Item1, tuple2.Item2].copy();

                    np_x[tuple2.Item1, tuple2.Item2] = tmp_val + eps;
                    var y1 = f.BaseForward([new Variable(new NDarray(np_x)), .. args]);
                    var y1arr = y1[0].Data.NumpyNDarray.copy();

                    np_x[tuple2.Item1, tuple2.Item2] = tmp_val - eps;
                    var y2 = f.BaseForward([new Variable(new NDarray(np_x)), .. args]);
                    var y2arr = y2[0].Data.NumpyNDarray.copy();

                    var diff = (y1arr - y2arr).sum();
                    grad[tuple2.Item1, tuple2.Item2] = diff / (2 * eps);

                    np_x[tuple2.Item1, tuple2.Item2] = tmp_val;
                    it.iternext();
                }
                args.ToList().ForEach(x => x.Data.Pop());
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
            var (na, nb) = Gpu.Available && Gpu.Use ? (cpExtensions.asnumpy(a.CupyNDarray), cpExtensions.asnumpy(b.CupyNDarray)) : (a.NumpyNDarray, b.NumpyNDarray);
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
