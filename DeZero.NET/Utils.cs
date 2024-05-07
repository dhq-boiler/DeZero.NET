using Cupy;
using DeZero.NET.Core;
using DeZero.NET.Functions;
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

        public static bool gradient_check(Function f, Params args, double rtol = 1e-4, double atol = 1e-5)
        {
            var x = args.Get<Variable>(0);

            x.Data = x.Data.astype(xp.float64);

            var num_grad = numerical_grad(f, Params.Base(args).SetKeywordArg(x));

            var y = f.Call(Params.Base(args).SetPositionalArgs(x));
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

        public static NDarray numerical_grad(Function f, Params args)
        {
            NDarray _x = args.Get<Variable>("x").Data;

            List<Core.Parameter> argsList = [..args.Through()];

            argsList.ForEach(x =>
            {
                if (Gpu.Available && Gpu.Use)
                {
                    x.Variable.Data.Switch(deleteOriginal:false);
                }
            });

            var eps = 1e-4;
            Numpy.NDarray np_x = _x.ToNumpyNDarray;

            if (Gpu.Available && Gpu.Use)
            {
                Gpu.Use = false;
                //argsList.ForEach(x => x.Variable.Data.Push(ArrayMode.np));
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
                    var tmp_val = _x.NumpyNDarray[idx].copy();

                    _x.NumpyNDarray[idx] = tmp_val + eps;
                    var x = new Variable(new NDarray(_x.NumpyNDarray, false));
                    var y1 = f.Call(
                        Params.Base(args).SetPositionalArgs(x));
                    var y1arr = y1[0].Data.NumpyNDarray.copy();

                    f.ResetParams();

                    _x.NumpyNDarray[idx] = tmp_val - eps; 
                    x = new Variable(new NDarray(_x.NumpyNDarray, false));
                    var y2 = f.Call(
                        Params.Base(args).SetPositionalArgs(x));
                    var y2arr = y2[0].Data.NumpyNDarray.copy();

                    var diff = (y1arr - y2arr).sum();
                    grad[idx] = diff / (2 * eps);

                    _x.NumpyNDarray[idx] = tmp_val;
                    it.iternext();
                }
                //argsList.ForEach(x => x.Variable.Data.Pop());
                Gpu.Use = true;
                return new NDarray(grad);
            }
            else
            {
                //argsList.ForEach(x => x.Variable.Data.Push(ArrayMode.np));
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
                    var tmp_val = _x.NumpyNDarray[idx].copy();

                    _x.NumpyNDarray[idx] = tmp_val + eps;
                    var x = new Variable(new NDarray(_x.NumpyNDarray, false));
                    var y1 = f.Call(
                        Params.Base(args).SetPositionalArgs(x));
                    var y1arr = y1[0].Data.NumpyNDarray.copy();

                    f.ResetParams();

                    _x.NumpyNDarray[idx] = tmp_val - eps;
                    x = new Variable(new NDarray(_x.NumpyNDarray, false));
                    var y2 = f.Call(
                        Params.Base(args).SetPositionalArgs(x));
                    var y2arr = y2[0].Data.NumpyNDarray.copy();

                    var diff = (y1arr - y2arr).sum();
                    grad[idx] = diff / (2 * eps);

                    _x.NumpyNDarray[idx] = tmp_val;
                    it.iternext();
                }
                //argsList.ForEach(x => x.Variable.Data.Pop());
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

        public static Variable im2col_array(Variable img, (int, int) kernel_size, (int, int) stride, (int, int) pad, bool to_matrix = true)
        {
            int N = img.Shape[0], C = img.Shape[1], H = img.Shape[2], W = img.Shape[3];
            int KH = kernel_size.Item1, KW = kernel_size.Item2;
            int SH = stride.Item1, SW = stride.Item2;
            int PH = pad.Item1, PW = pad.Item2;
            var OH = Utils.get_conv_outsize(H, KH, SH, PH);
            var OW = Utils.get_conv_outsize(W, KW, SW, PW);

            Variable ret = null;

            if (Gpu.Available && Gpu.Use)
            {
                var col = _im2col_gpu(img, kernel_size, stride, pad);
                ret = new NDarray(col).ToVariable();
            }
            else
            {
                var _img = np.pad(img.Data.ToNumpyNDarray, xp.array([[0, 0], [0, 0], [PH, PH + SH - 1], [PW, PW + SW - 1]]).ToNumpyNDarray, "constant", constant_values: [0]);
                var col = np.zeros(new Numpy.Models.Shape(N, C, KH, KW, OH, OW), dtype: img.Dtype.NumpyDtype);
                var colSlice = new Numpy.Models.Slice(null, null);

                foreach (var j in Enumerable.Range(0, KH))
                {
                    var j_lim = j + SH * OH;
                    var imgSlice1 = new Numpy.Models.Slice(j, j_lim, SH);
                    foreach (var i in Enumerable.Range(0, KW))
                    {
                        var i_lim = i + SW * OW;
                        var imgSlice2 = new Numpy.Models.Slice(i, i_lim, SW);
                        col[colSlice, colSlice, j, i, colSlice, colSlice] = _img[colSlice, colSlice, imgSlice1, imgSlice2];
                    }
                }

                ret = new NDarray(col).ToVariable();
            }

            if (to_matrix)
            {
                ret = Reshape.Invoke(Transpose.Invoke(ret, [new Axis([0, 4, 5, 1, 2, 3])])[0], new Shape(N * OH * OW, -1))[0];
            }

            return ret;
        }

        private static int get_conv_outsize(int input_size, int kernel_size, int stride, int pad)
        {
            return (int)(input_size + pad * 2 - kernel_size) / (int)(stride) + 1;
        }

        private static Cupy.NDarray _im2col_gpu(Variable img, (int, int) kernel_size, (int, int) stride, (int, int) pad)
        {
            int n = img.Shape[0], c = img.Shape[1], h = img.Shape[2], w = img.Shape[3];
            int kh = kernel_size.Item1, kw = kernel_size.Item2;
            int sy = stride.Item1, sx = stride.Item2;
            int ph = pad.Item1, pw = pad.Item2;
            int out_h = Utils.get_conv_outsize(h, kh, sy, ph);
            int out_w = Utils.get_conv_outsize(w, kw, sx, pw);
            int dy = 1, dx = 1;
            var col = cp.empty(new Cupy.Models.Shape(n, c, kh, kw, out_h, out_w), dtype: img.Dtype.CupyDtype);

            cp.ElementwiseKernel<PyObject, PyObject, int, int, int, int, int, int, int, int, int, int, int, int, PyObject>(
                "raw T img, int32 h, int32 w, int32 out_h, int32 out_w,"
               + "int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,"
               + "int32 dy, int32 dx",
                "T col",
                """
                int c0 = i / (kh * kw * out_h * out_w);
                int ky = i / (kw * out_h * out_w) % kh;
                int kx = i / (out_h * out_w) % kw;
                int out_y = i / out_w % out_h;
                int out_x = i % out_w;
                int in_y = ky * dy + out_y * sy - ph;
                int in_x = kx * dx + out_x * sx - pw;
                if (in_y >= 0 && in_y < h && in_x >= 0 && in_x < w)
                {
                    col = img[in_x + w * (in_y + h * c0)];
                }
                else
                {
                    col = 0;
                }
                """, img.Data.reduced_view().CupyNDarray.PyObject,
                h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, dy, dx, col.PyObject,
                name: "im2col");

            return new Cupy.NDarray(col);
        }

        internal static int get_deconv_outsize(int size, int k, int s, int p)
        {
            return s * (size - 1) + k - 2 * p;
        }

        public static NDarray col2im_array(NDarray col, (int, int, int, int) imgShape, (int, int) kernel_size, (int, int) stride, (int, int) pad, bool to_matrix = true)
        {
            int N = imgShape.Item1, C = imgShape.Item2, H = imgShape.Item3, W = imgShape.Item4;
            int KH = kernel_size.Item1, KW = kernel_size.Item2;
            int SH = stride.Item1, SW = stride.Item2;
            int PH = pad.Item1, PW = pad.Item2;
            int OH = Utils.get_conv_outsize(H, KH, SH, PH);
            int OW = Utils.get_conv_outsize(W, KW, SW, PW);

            if (to_matrix)
            {
                col = col.reshape(N, OH, OW, C, KH, KW).transpose(0, 3, 4, 5, 1, 2);
            }

            if (Gpu.Available && Gpu.Use)
            {
                var img = Utils._col2im_gpu(col, SH, SW, PH, PW, H, W);
                return new NDarray(img);
            }
            else
            {
                var img = np.zeros(new Numpy.Models.Shape(N, C, H + 2 * PH + SH - 1, W + 2 * PW + SW - 1), dtype: col.dtype.NumpyDtype);
                var colSlice = new Numpy.Models.Slice(null, null);
                foreach (var j in Enumerable.Range(0, KH))
                {
                    var j_lim = j + SH * OH;
                    var imgSlice1 = new Numpy.Models.Slice(j, j_lim, SH);
                    foreach (var i in Enumerable.Range(0, KW))
                    {
                        var i_lim = i + SW * OW;
                        var imgSlice2 = new Numpy.Models.Slice(i, i_lim, SW);
                        img[colSlice, colSlice, imgSlice1, imgSlice2] += col.NumpyNDarray[colSlice, colSlice, j, i, colSlice, colSlice];
                    }
                }
                return new NDarray(img[colSlice, colSlice, new Numpy.Models.Slice(PH, H+PH), new Numpy.Models.Slice(PW, W + PW)]);
            }
        }

        private static Cupy.NDarray _col2im_gpu(NDarray col, int sy, int sx, int ph, int pw, int h, int w)
        {
            int n = col.shape[0], c = col.shape[1], kh = col.shape[2], kw = col.shape[3], out_h = col.shape[4], out_w = col.shape[5];
            int dx = 1, dy = 1;
            var img = cp.empty(new Cupy.Models.Shape(n, c, h, w), dtype: col.dtype.CupyDtype);

            cp.ElementwiseKernel<PyObject, PyObject, int, int, int, int, int, int, int, int, int, int, int, int, PyObject>(
                "raw T col, int32 h, int32 w, int32 out_h, int32 out_w,"
                                          + "int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,"
                                          + "int32 dx, int32 dy",
                "T img",
                """
                int c0 = i / (h * w);
                int y = i / w % h;
                int x = i % w;
                T val = 0;
                for (int ky = 0; ky < kh; ++ky)
                {
                    int out_y = (y + ph - ky * dy);
                    if (0 > out_y || out_y >= out_h * sy) continue;
                    if (out_y % sy != 0) continue;
                    out_y /= sy;
                    for (int kx = 0; kx < kw; ++kx)
                    {
                        int out_x = (x + pw - kx * dx);
                        if (0 > out_x || out_x >= out_w * sx) continue;
                        if (out_x % sx != 0) continue;
                        out_x /= sx;
                        int k = out_y + out_h * (kx + kw * (ky + kh * c0));
                        val = val + col[out_x + out_w * k];
                    }
                }
                img = val;
                """, col.reduced_view().CupyNDarray.PyObject, h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, dx, dy, img.PyObject,
                name: "col2im");

            return img;
        }

        public static Variable conv2d_simple(Variable x, Variable W, Variable b = null, (int, int)? stride = null, (int, int)? pad = null)
        {
            if (!stride.HasValue)
            {
                stride = (1, 1);
            }

            if (!pad.HasValue)
            {
                pad = (0, 0);
            }

            var x_val = x;
            var W_val = W;
            var Weight = W;
            int N = x_val.Shape[0], C = x_val.Shape[1], H = x_val.Shape[2], _W = x_val.Shape[3];
            int OC = Weight.Shape[0], C_ = Weight.Shape[1], KH = Weight.Shape[2], KW = Weight.Shape[3];
            int SH = stride.Value.Item1, SW = stride.Value.Item2;
            int PH = pad.Value.Item1, PW = pad.Value.Item2;
            int OH = Utils.get_conv_outsize(H, KH, SH, PH);
            int OW = Utils.get_conv_outsize(_W, KW, SW, PW);

            var col = Utils.im2col(x, (KH, KW), stride, pad, to_matrix: true);
            W_val = Transpose.Invoke(Reshape.Invoke(W_val, new Shape(OC, -1))[0])[0];
            var t = Linear.Invoke(col, W_val, b)[0];
            var y = Transpose.Invoke(Reshape.Invoke(t, new Shape(N, OH, OW, OC))[0], [new Axis([0, 3, 1, 2])])[0];
            return y;
        }

        public static Variable conv2d_simple(Variable x, Variable W, Variable b = null, int stride = 1, int pad = 0)
        {
            var x_val = x;
            var W_val = W;
            var Weight = W;
            int N = x_val.Shape[0], C = x_val.Shape[1], H = x_val.Shape[2], _W = x_val.Shape[3];
            int OC = Weight.Shape[0], C_ = Weight.Shape[1], KH = Weight.Shape[2], KW = Weight.Shape[3];
            int SH = stride, SW = stride;
            int PH = pad, PW = pad;
            int OH = Utils.get_conv_outsize(H, KH, SH, PH);
            int OW = Utils.get_conv_outsize(_W, KW, SW, PW);

            var col = Utils.im2col(x, (KH, KW), (stride, stride), (pad, pad), to_matrix: true);
            W_val = Transpose.Invoke(Reshape.Invoke(W_val, new Shape(OC, -1))[0])[0];
            var t = Linear.Invoke(col, W_val, b)[0];
            var y = Transpose.Invoke(Reshape.Invoke(t, new Shape(N, OH, OW, OC))[0], [new Axis([0, 3, 1, 2])])[0];
            return y;
        }

        private static Variable im2col(Variable x, (int KH, int KW) kernel_size, (int, int)? stride, (int, int)? pad, bool to_matrix = true)
        {
            if (!stride.HasValue)
            {
                stride = (1, 1);
            }

            if (!pad.HasValue)
            {
                pad = (0, 0);
            }
            var y = Im2col.Invoke(x, kernel_size, stride, pad, to_matrix);
            return y;
        }
    }
}
