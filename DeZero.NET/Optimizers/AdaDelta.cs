using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Optimizers
{
    public class AdaDelta : Optimizer
    {
        public float eps { get; set; }
        public float rho { get; set; }
        public Property<Dictionary<int, Variable>> msg { get; } = new(nameof(msg));
        public Property<Dictionary<int, Variable>> msdx { get; } = new(nameof(msdx));

        public AdaDelta(float rho = 0.95f, float eps = 1e-6f) : base()
        {
            this.rho = rho;
            this.eps = eps;
            this.msg.Value = new Dictionary<int, Variable>();
            this.msdx.Value = new Dictionary<int, Variable>();
            RegisterNonVolatileParameters(this.msg, this.msdx);
        }

        public override void UpdateOne(Parameter param)
        {
            var key = param.GetHashCode();
            if (!msg.Value.ContainsKey(key))
            {
                msg.Value[key] = xp.zeros_like(param.Data.Value).ToVariable();
                msdx.Value[key] = xp.zeros_like(param.Data.Value).ToVariable();
            }

            var _msg = msg.Value[key];
            var _msdx = msdx.Value[key];
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
