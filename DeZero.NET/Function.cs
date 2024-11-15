using DeZero.NET.Core;
using DeZero.NET.Extensions;
using System.Diagnostics;

namespace DeZero.NET
{
    public class Function
    {
        internal long _ForwardedTicks = DateTime.MinValue.Ticks;
        protected Func<Params, Variable[]> _f;
        public int Generation { get; set; }
        public IEnumerable<Core.Parameter> Inputs { get; private set; }
        public IEnumerable<Variable> Outputs { get; private set; }

        [DebuggerStepThrough]
        protected Function()
        {
        }

        [DebuggerStepThrough]
        public Function(Func<Params, Variable[]> f)
        {
            _f = f;
        }

        public virtual Variable[] Call(Params args)
        {
            using var scope = new BatchScope();

            var ys = Forward(args);
            var outputs = ys.Select(y => (xp.isscalar(y.Data.Value) ? xp.array(y.Data.Value).ToVariable() : y)).ToList();

            if (Config.EnableBackprop)
            {
                foreach (var output in outputs)
                {
                    if (this.GetType().Name != "Function")
                    {
                        output.Creator = this;
                    }
                    scope.PreserveVariable(output);
                }

                // 古い入力をクリーンアップ
                if (this.Inputs != null)
                {
                    foreach (var input in Inputs)
                    {
                        scope.TrackTemporary(input.Variable);
                    }
                }

                this.Inputs = args.Through;
                int gen = Generation;
                foreach (var input in Inputs)
                {
                    input.Variable.Generation = ++gen;
                }

                // 古い出力をクリーンアップ
                if (this.Outputs != null)
                {
                    foreach (var output in Outputs)
                    {
                        scope.TrackTemporary(output);
                    }
                }

                this.Outputs = outputs;
                Generation = Inputs.Select(x => x.Variable.Generation).Max() + 1;

                if (GpuMemoryMonitor.Instance.GetCurrentMemoryUsage() > 200) // MB
                {
                    GpuMemoryMonitor.ForceMemoryPool();
                }
            }

            return outputs.ToArray();
        }

        //public virtual Variable[] Call(Params args)
        //{
        //    var ys = Forward(args);

        //    var outputs = ys.Select(y => (xp.isscalar(y.Data.Value) ? xp.array(y.Data.Value).ToVariable() : y)).ToList();

        //    if (Config.EnableBackprop)
        //    {
        //        foreach (var output in outputs)
        //        {
        //            if (this.GetType().Name != "Function")
        //            {
        //                output.Creator = this;
        //            }

        //            //if (this.Inputs is not null)
        //            //{
        //            //    foreach (var input in Inputs)
        //            //    {
        //            //        if (input.Variable is not null)
        //            //        {
        //            //            input.Variable.Dispose();
        //            //        }
        //            //    }

        //            //    this.Inputs = null;
        //            //}
        //            this.Inputs = args.Through;
        //            int gen = Generation;
        //            foreach (var input in Inputs)
        //            {
        //                input.Variable.Generation = ++gen;
        //            }
        //        }

        //        //if (this.Outputs is not null)
        //        //{
        //        //    foreach (var _output in Outputs)
        //        //    {
        //        //        _output.Dispose();
        //        //    }
        //        //    this.Outputs = null;
        //        //}
        //        this.Outputs = outputs;

        //        Generation = Inputs.Select(x => x.Variable.Generation).Max() + 1;
        //    }

        //    return outputs.ToArray();
        //}

        public virtual Variable[] Forward(Params args)
        {
            return _f(args);
        }

        public virtual Variable[] Backward(Params args)
        {
            return args.Through.Select(x => x.Variable).ToArray();
        }

        public override int GetHashCode()
        {
            return base.GetHashCode() ^ Generation.GetHashCode();
        }

        public virtual void ResetParams()
        {
        }
    }
}
