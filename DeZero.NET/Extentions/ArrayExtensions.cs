using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeZero.NET.Extentions
{
    public static class ArrayExtensions
    {
        public static int Product(this int[] array)
        {
            if (array == null || array.Length == 0)
                return 0;

            int result = 1;
            foreach (int num in array)
            {
                result *= num;
            }
            return result;
        }
    }
}
