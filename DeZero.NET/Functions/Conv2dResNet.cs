using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class Conv2dResNet : Functions.Conv2d
    {
        public Conv2dResNet(int stride, int pad) : base(stride, pad)
        {
        }

        public Conv2dResNet((int, int) stride, (int, int) pad) : base(stride, pad)
        {
        }

        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>("x");
            var W = args.Get<Variable>("W");
            var b = args.Get<Variable>("b");

            using var x_shape = x.Shape;
            using var W_shape = W.Shape;
            Shape KH = W_shape[2], KW = W_shape[3];
            var col = Utils.im2col_array(x, (KH[0], KW[0]), Stride, Pad, to_matrix: false);

            var colShape = col.Shape;
            NDarray y = default;

            using var shape = col.Data.Value.shape;
            using var shape1 = new Shape(x_shape[0], 1, 3, 3, 28, 28);
            using var shape2 = new Shape(x_shape[0], 16, 3, 3, 28, 28);
            using var shape3 = new Shape(x_shape[0], 16, 1, 1, 28, 28);
            using var shape4 = new Shape(x_shape[0], 32, 3, 3, 28, 28);
            using var shape5 = new Shape(x_shape[0], 32, 1, 1, 28, 28);
            using var shape6 = new Shape(x_shape[0], 64, 3, 3, 28, 28);
            if (shape == shape1)
            {
                y = xp.tensordot(col.Data.Value, W.Data.Value, [[2, 3], [2, 3]]);
                y = y.reshape(x_shape[0], 16, 28, 28);
            }
            else if (shape == shape2)
            {
                y = xp.tensordot(col.Data.Value, W.Data.Value, [[1, 2, 3], [1, 2, 3]]);

                using var shape_a = new Shape(16, 16, 3, 3);
                using var shape_b = new Shape(32, 16, 3, 3);
                if (W_shape == shape_a)
                {
                    y = y.reshape(x_shape[0], 16, 28, 28);
                }
                else if (W_shape == shape_b)
                {
                    y = y.reshape(x_shape[0], 32, 28, 28);
                }
            }
            else if (shape == shape3)
            {

            }
            else if (shape == shape4)
            {
                using var shape_a = new Shape(32, 32, 3, 3);
                using var shape_b = new Shape(32, 16, 3, 3);
                using var shape_c = new Shape(64, 32, 3, 3);
                if (W_shape == shape_a)
                {
                    y = xp.tensordot(col.Data.Value, W.Data.Value, [[1, 2, 3], [1, 2, 3]]);
                    y = y.reshape(x_shape[0], 32, 28, 28);
                }
                else if (W_shape == shape_b)
                {
                    y = xp.tensordot(col.Data.Value, W.Data.Value, [[1, 2, 3], [0, 2, 3]]);
                    y = y.reshape(x_shape[0], 16, 28, 28);
                }
                else if (W_shape == shape_c)
                {
                    y = xp.tensordot(col.Data.Value, W.Data.Value, [[1, 2, 3], [1, 2, 3]]);
                    y = y.reshape(x_shape[0], 64, 28, 28);
                }
            }
            else if (shape == shape5)
            {
                y = xp.tensordot(col.Data.Value, W.Data.Value, [[1, 2, 3], [1, 2, 3]]);
                if (W_shape[0] == 32)
                {
                    y = y.reshape(x_shape[0], 32, 28, 28);
                }
                else if (W_shape[0] == 64)
                {
                    y = y.reshape(x_shape[0], 64, 28, 28);
                }
            }
            else if (shape == shape6)
            {
                using var shape_a = new Shape(64, 32, 3, 3);
                using var shape_b = new Shape(64, 64, 3, 3);
                if (W_shape == shape_a)
                {
                    y = xp.tensordot(col.Data.Value, W.Data.Value, [[1, 2, 3], [0, 2, 3]]);
                    y = y.reshape(x_shape[0], 32, 28, 28);
                }
                else if (W_shape == shape_b)
                {
                    y = xp.tensordot(col.Data.Value, W.Data.Value, [[1, 2, 3], [1, 2, 3]]);
                    y = y.reshape(x_shape[0], 64, 28, 28);
                }
            }

            return [y.Relay(this)];
        }

        public new static Variable[] Invoke(Variable x, Variable W, Variable b = null, (int, int)? stride = null, (int, int)? pad = null)
        {
            if (!stride.HasValue)
            {
                stride = (1, 1);
            }
            if (!pad.HasValue)
            {
                pad = (0, 0);
            }

            // チャンネル数の変更に対応
            using var x_shape = x.Shape;
            using var W_shape = W.Shape;
            if (x_shape[1] != W_shape[1])
            {
                var W_proj = xp.zeros(new Shape(W_shape[0], x_shape[1], 3, 3));
                var conv_proj = new Conv2dResNet((1, 1), (1, 1)).Call(Params.New.SetKeywordArg(x, W_proj, default(Variable), "x", "W", "b"));
                x = conv_proj[0];
            }

            // ストライドの処理
            if (stride.Value.Item1 == 2)
            {
                x = new Conv2dResNet(stride.Value, pad.Value).Call(Params.New.SetKeywordArg(x, W, b))[0];
            }
            else
            {
                x = new Conv2dResNet(stride.Value, pad.Value).Call(Params.New.SetKeywordArg(x, W, b))[0];
            }

            return [x];
        }

        //public static NDarray _convolve2d(NDarray image, NDarray kernel)
        //{
        //    var shape = new Shape(image.shape[0] - kernel.shape[0] + 1, image.shape[1] - kernel.shape[1] + 1) + kernel.shape;
        //    var strides = xp.array(image.strides) * 2;
        //    var strided_image = xp.lib.stride_tricks.as_strided(image, kernel.shape, strides.GetData<int[]>());
        //    return xp.einsum("ij,ij->ij", kernel, strided_image);
        //}

        //public static NDarray _convolve2d_multichannel(NDarray image, NDarray kernel)
        //{
        //    var convolved_image = xp.empty(new Shape(image.shape[0] - kernel.shape[0] + 1, image.shape[1] - kernel.shape[1] + 1,
        //        image.shape[2]));
        //    foreach (var i in Enumerable.Range(0, image.shape[2]))
        //    {
        //        convolved_image[new Slice(), new Slice(), i] = _convolve2d(image[new Slice(), new Slice(), i], kernel);
        //    }
        //    return convolved_image;
        //}

        //public static NDarray _pad_singlechannel_image(NDarray image, Shape kernel_shape, string mode)
        //{
        //    return xp.pad(image, xp.array([[kernel_shape[0] / 2], [kernel_shape[1] / 2]]), mode);
        //}

        //public static NDarray _pad_multichannel_image(NDarray image, Shape kernel_shape, string mode)
        //{
        //    return xp.pad(image, xp.array([[kernel_shape[0] / 2], [kernel_shape[1] / 2], [0]]), mode);
        //}

        //public static NDarray convolve2d(NDarray image, NDarray kernel, string mode)
        //{
        //    if (image.ndim == 2)
        //    {
        //        var pad_image = mode is not null ? _pad_singlechannel_image(image, kernel.shape, mode) : image;
        //        return _convolve2d(pad_image, kernel);
        //    }
        //    else if (image.ndim == 3)
        //    {
        //        var pad_image = mode is not null ? _pad_multichannel_image(image, kernel.shape, mode) : image;
        //        return _convolve2d_multichannel(pad_image, kernel);
        //    }
        //    else
        //    {
        //        throw new Exception("Invalid image shape");
        //    }
        //}
    }
}
