using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeZero.NET.Log
{
    public interface IProgressLogger : IDisposable
    {
        void Complete(string message = "Completed");
        void Failed(string message);
    }
}
