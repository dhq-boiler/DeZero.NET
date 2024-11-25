using DeZero.NET.Core;
using DeZero.NET.Extensions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeZero.NET.Layers
{
    public class ZeroPadding : Layer
    {
        private int[] _padding;
        private Shape _originalShape;

        public ZeroPadding(int[] padding)
        {

            if (padding.Length == 2)
            {
                _padding = padding;
            }
            else if (padding.Length == 4)
            {
                _padding = padding;
            }
            else
            {
                throw new ArgumentException("Invalid padding dimensions.");
            }
        }

        public override Variable[] Forward(params Variable[] xs)
        {
            var x = xs[0];
            _originalShape = x.Shape;
            var paddedX = PadForward(x, _padding);
            return [paddedX];
        }

        public override Variable[] Backward(params Variable[] gys)
        {
            var gy = gys[0];
            var paddedGy = PadBackward(gy, _padding);
            return [paddedGy];
        }

        private Variable PadForward(Variable x, int[] padding)
        {
            var paddedX = xp.pad(x.Data.Value, xp.array(padding), "empty").ToVariable();
            return paddedX;
        }

        private Variable PadBackward(Variable gy, int[] padding)
        {
            var gx = RemovePadding(gy.Data.Value, _originalShape.Dimensions, padding);
            return gx.ToVariable();
        }

        // 1次元配列に対するパディングの削除例
        public static NDarray RemovePadding(NDarray gradArray, int[] originalShape, int[] padding)
        {
            using var gradArray_shape = gradArray.shape;
            // 各次元に対してパディングを削除
            int startRow = padding[0];
            int endRow = gradArray_shape[0] - padding[1];
            int startCol = padding[2];
            int endCol = gradArray_shape[1] - padding[3];

            using var rowSlice = new Slice(startRow, endRow);
            using var colSlice = new Slice(startCol, endCol);
            // スライスを使用してパディングを削除
            var slicedArray = gradArray[rowSlice, colSlice];

            // 必要に応じて形状を調整
            slicedArray = slicedArray.reshape(originalShape);
            return slicedArray;
        }
    }
}
