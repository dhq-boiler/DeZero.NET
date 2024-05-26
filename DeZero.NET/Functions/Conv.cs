using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeZero.NET.Functions
{
    public static class Conv
    {
        public static Variable pooling_simple(Variable x, int kernel_size, int stride = 1, int pad = 0)
        {
            int N = x.Shape[0], C = x.Shape[1], H = x.Shape[2], W = x.Shape[3];
            int KH = kernel_size, KW = kernel_size;
            int PH = pad, PW = pad;
            int SH = stride, SW = stride;
            int OH = Utils.get_conv_outsize(H, KH, SH, PH);
            int OW = Utils.get_conv_outsize(W, KW, SW, PW);

            var col = Utils.im2col(x, (KH, KW), (stride, stride), (pad, pad), to_matrix: true);
            col = Reshape.Invoke(col, new Shape(-1, KH * KW))[0];
            var y = Max.Invoke(col, axis: [1])[0];
            y = Transpose.Invoke(Reshape.Invoke(y, new Shape(N, OH, OW, C))[0], [0, 3, 1, 2])[0];
            return y;
        }
    }
}
