using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DeZero.NET.Functions;

namespace DeZero.NET
{
    public static class BroadcastUtils
    {
        /// <summary>
        /// ブロードキャストされた勾配を元の形状に戻します。
        /// </summary>
        /// <param name="gy">ブロードキャストされた勾配</param>
        /// <param name="originalShape">元の形状</param>
        /// <returns>調整された勾配</returns>
        public static Variable SumToShape(Variable gy, Shape originalShape)
        {
            var currentShape = gy.Shape;
            if (originalShape.Equals(currentShape))
            {
                return gy;
            }

            // 次元数を合わせる
            while (currentShape.ndim > originalShape.ndim)
            {
                gy = Sum.Invoke(gy, axis: 0)[0];
                currentShape = gy.Shape;
            }

            // 各次元でサイズが1の場合は合計を取る
            for (int axis = 0; axis < originalShape.ndim; axis++)
            {
                if (originalShape[axis] == 1 && currentShape[axis] != 1)
                {
                    gy = Sum.Invoke(gy, axis: axis, keepdims: true)[0];
                }
            }

            return gy;
        }
    }
}
