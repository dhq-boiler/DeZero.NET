﻿using DeZero.NET.Core;
using DeZero.NET.Extensions;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using Amazon.Runtime.EventStreams.Internal;

namespace DeZero.NET
{
    public class Function
    {
        internal long _ForwardedTicks = DateTime.MinValue.Ticks;
        protected Func<Params, Variable[]> _f;
        public string Id { get; } = Guid.NewGuid().ToString().Substring(0, 6);
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
            var outputs = ys.Select(y => (xp.isscalar(y.Data.Value) ? xp.array(y.Data.Value).ToVariable(y, this).copy() : y.Relay(this).copy())).ToList();

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

                this.Inputs = args.Through.OfType<Core.Parameter>().Where(x => x.Variable is not null)
                    .Select(x =>
                    {
                        if (x is Core.Parameter pa && pa.Variable?.Data?.Value is null)
                        {
                            return null;
                        }
                        var parameter = x;
                        parameter.Value = parameter.Variable.copy();
                        return parameter;
                    })
                    .Union(args.Through.OfType<Core.Parameter>().Where(x => x.Value is IEnumerable<DeZero.NET.Parameter>)
                        .SelectMany(x => x.Value as IEnumerable<DeZero.NET.Parameter>).Select(x =>
                {
                    if (x is DeZero.NET.Parameter pa && pa?.Data?.Value is null)
                    {
                        return null;
                    }
                    Variable parameter = x;
                    parameter = parameter.copy();
                    return new Core.Parameter(null, parameter);
                })).Where(x => x is not null).ToList();



                int gen = Generation;
                foreach (var input in Inputs)
                {
                    input.Variable.Generation = ++gen;
                }

                //古い出力をクリーンアップ
                if (this.Outputs != null)
                {
                    foreach (var output in Outputs)
                    {
                        scope.TrackTemporary(output);
                    }
                }

                this.Outputs = outputs.Select(x =>
                {
                    if (x is Variable pa && pa?.Data?.Value is null)
                    {
                        return null;
                    }

                    var newx = x.copy(holdReference: false);
                    newx.Grad.SetValueWithNoFireEvent(x.Grad.Value?.copy(holdReference: false));
                    return newx;
                }).Where(x => x is not null).ToList();
                Generation = Inputs.Select(x => x.Variable.Generation).Max() + 1;

                if (GpuMemoryMonitor.Instance.GetCurrentMemoryUsage() > 200) // MB
                {
                    GpuMemoryMonitor.ForceMemoryPool();
                }
            }

            return outputs.ToArray();
        }

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
            return Id.GetHashCode() ^ Generation.GetHashCode();
        }

        public virtual void ResetParams()
        {
        }
    }
}
