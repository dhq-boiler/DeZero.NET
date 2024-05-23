using DeZero.NET.Core;
using DeZero.NET.Models;
using DeZero.NET.Optimizers.HookFunctions;

namespace DeZero.NET.Optimizers
{
    public abstract class Optimizer
    {
        public Model Target { get; set; }

        public List<HookFunction> Hooks { get; set; }

        public Optimizer()
        {
            this.Target = null;
            this.Hooks = new List<HookFunction>();
        }

        public Optimizer Setup(Model target)
        {
            this.Target = target;
            return this;
        }

        public virtual void Update(Params args)
        {
            var _params = Target.Params().Where(p => p.Grad.Value is not null).ToArray();

            foreach (var f in Hooks)
            {
                f.Call(_params);
            }

            foreach (var param in _params)
            {
                UpdateOne(param);
            }
        }

        public abstract void UpdateOne(Parameter param);

        public void AddHook(HookFunction f)
        {
            Hooks.Add(f);
        }
    }
}
