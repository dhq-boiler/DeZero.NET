using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeZero.NET.Optimizers
{
    public class AdaDelta : Optimizer
    {
        public float eps { get; set; }
        public float rho { get; set; }
        public Dictionary<int, Variable> msg { get; set; }
        public Dictionary<int, Variable> msdx { get; set; }

        public AdaDelta(float rho = 0.95f, float eps = 1e-6f) : base()
        {
            this.rho = rho;
            this.eps = eps;
            this.msg = new Dictionary<int, Variable>();
            this.msdx = new Dictionary<int, Variable>();
        }

        public override void UpdateOne(Parameter param)
        {
            var key = param.GetHashCode();
            if (!msg.ContainsKey(key))
            {
                msg[key] = xp.zeros_like(param.Data.Value).ToVariable();
                msdx[key] = xp.zeros_like(param.Data.Value).ToVariable();
            }

            var _msg = msg[key];
            var _msdx = msdx[key];
            var rho = this.rho;
            var eps = this.eps;
            var grad = param.Grad.Value.Data.Value;

            _msg *= rho;
            _msg += (1 - rho) * grad * grad;
            var dx = xp.sqrt((_msdx.Data.Value + eps) / (_msg.Data.Value + eps)) * grad;

            _msdx *= rho;
            _msdx += (1 - rho) * dx * dx;
            param.Data.Value -= dx;
        }

    }
}
