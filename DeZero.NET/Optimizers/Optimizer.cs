using DeZero.NET.Core;
using DeZero.NET.Models;

namespace DeZero.NET.Optimizers
{
    public abstract class Optimizer
    {
        public Model Target { get; set; }

        public List<Function> Hooks { get; set; }

        public Optimizer()
        {
            this.Target = null;
            this.Hooks = new List<Function>();
        }

        public Optimizer Setup(Model target)
        {
            this.Target = target;
            return this;
        }

        public virtual void Update(Params args)
        {
            var _params = Target.Params().Where(p => p.Grad is not null).ToArray();

            foreach (var f in Hooks)
            {
                f.Call(Core.Params.New.AddRange(_params));
            }

            foreach (var param in _params)
            {
                UpdateOne(param);
            }
        }

        public abstract void UpdateOne(Parameter param);

        public void AddHook(Function f)
        {
            Hooks.Add(f);
        }
    }
}
