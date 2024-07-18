using Python.Runtime;
using System.Text.RegularExpressions;

namespace DeZero.NET.Core
{
    public static class IndexConverter
    {
        public static int[] ConvertPyObjectToIntArray(PyObject pyObject)
        {
            // PyObjectがタプルまたはリストの場合、その要素をint[]に変換
            if (pyObject.IsIterable())
            {
                // PyObjectをPythonのリストに変換（タプルの場合もリストになる）
                using PyTuple list = pyObject.As<PyTuple>();

                // リストの要素数を取得
                long length = list.Length();
                int[] result = new int[length];

                for (int i = 0; i < length; i++)
                {
                    // 各要素をintに変換して配列に格納
                    using (PyObject item = list[i])
                    {
                        result[i] = item.As<int>();
                    }
                }

                return result;
            }
            else
            {
                // PyObjectが単一の整数の場合
                if (Regex.IsMatch(pyObject.ToString(), @"$\d+?^"))
                {
                    return new int[] { pyObject.As<int>() };
                }
                // その他の型の場合は例外を投げる
                else
                {
                    throw new InvalidOperationException("PyObject is not iterable and not a single integer.");
                }
            }
        }

    }
}
