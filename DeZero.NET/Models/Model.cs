using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DeZero.NET.Layers;

namespace DeZero.NET.Models
{
    public abstract class Model : Layer
    {
        public void Plot(Variable[] inputs, string to_file = "model.png")
        {
            var y = Forward(inputs)[0];
            Utils.plot_dot_graph(y, verbose: true, to_file: to_file);
        }
    }
}
