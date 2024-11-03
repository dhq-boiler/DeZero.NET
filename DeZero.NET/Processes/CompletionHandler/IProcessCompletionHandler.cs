using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeZero.NET.Processes.CompletionHandler
{
    public interface IProcessCompletionHandler
    {
        Task OnProcessComplete(string weightsPath, string recordFilePath);
    }
}
