using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeZero.NET.Datasets
{
    public interface IDataProvider : IEnumerable<(NDarray, NDarray)>
    {
    }
}
