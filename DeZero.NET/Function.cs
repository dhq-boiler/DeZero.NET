using DeZero.NET.Core;
using DeZero.NET.Extensions;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using Amazon.Runtime.EventStreams.Internal;
using System.Linq;

namespace DeZero.NET
{
    public class Function
    {
        internal long _ForwardedTicks = DateTime.MinValue.Ticks;
        protected Func<Params, Variable[]> _f;
        public string Id { get; } = Guid.NewGuid().ToString().Substring(0, 6);
        public int Generation { get; set; }
        public IEnumerable<Core.Parameter> Inputs { get; internal set; }
        public IEnumerable<Variable> Outputs { get; internal set; }

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
            //using var scope = new BatchScope();

            //if (args.Through.Any(x => x is Core.Parameter p &&
            //                          p.Value is Variable &&
            //                          p.Variable is Variable v &&
            //                          v is not null &&
            //                          v.Shape is Shape s &&
            //                          s?.PyObject?.Handle != IntPtr.Zero && s.ToString() == "(384, 384, 3, 3)"))
            //{
            //    Debugger.Break();
            //}

            var ys = Forward(args);
            //GpuMemoryMonitor.Instance.LogMemoryUsage("outputs being");
            var outputs = ys.Select(y =>
            {
                if (xp.isscalar(y.Data.Value))
                {
                    return xp.array(y.Data.Value).ToVariable(y, this).copy();
                }
                else
                {
                    return y.Relay(this).copy();
                }
            }).ToList();
            //GpuMemoryMonitor.Instance.LogMemoryUsage("outputs end");
            //if (outputs.Any(x => x is not null && x.Shape.ToString() == "(384, 384, 3, 3)"))
            //{
            //    Debugger.Break();
            //}

            if (Config.EnableBackprop)
            {
                foreach (var output in outputs)
                {
                    if (this.GetType().Name != "Function")
                    {
                        output.Creator = this;
                    }
                    //scope.PreserveVariable(output);
                }

                // 古い入力をクリーンアップ
                if (this.Inputs != null)
                {
                    foreach (var input in Inputs.Where(x => x.Variable is Variable))
                    {
                        //scope.TrackTemporary(input.Variable);
                        input.Variable.Dispose();
                    }
                }

                //GpuMemoryMonitor.Instance.LogMemoryUsage("set Inputs begin");
                this.Inputs = args.Through.OfType<Core.Parameter>().Where(x => x.Variable is not null)
                    .Select(x =>
                    {
                        //GpuMemoryMonitor.Instance.LogMemoryUsage("if block begin");
                        if (x is Core.Parameter pa)
                        {
                            //GpuMemoryMonitor.Instance.LogMemoryUsage("if block 1");
                            if (pa.Value is not Variable)
                            {
                                return null;
                            }

                            //GpuMemoryMonitor.Instance.LogMemoryUsage("if block 2");
                            if (pa?.Variable is null)
                            {
                                return null;
                            }

                            //GpuMemoryMonitor.Instance.LogMemoryUsage("if block 3");
                            if (pa?.Variable.Data.Value.Handle == IntPtr.Zero)
                            {
                                return null;
                            }
                        }

                        //GpuMemoryMonitor.Instance.LogMemoryUsage("if block end");

                        //GpuMemoryMonitor.Instance.LogMemoryUsage("if block 4");
                        //var parameter = x;
                        //GpuMemoryMonitor.Instance.LogMemoryUsage("if block 5");
                        //parameter.Value = parameter.Variable.Relay(null, x.Variable).copy();
                        //GpuMemoryMonitor.Instance.LogMemoryUsage("if block 6");
                        //return parameter;
                        var _x = x.Variable.Relay(null, x.Variable).copy();
                        //if (x.Value is IDisposable disposable)
                        //{
                        //    disposable.Dispose();
                        //}
                        x.Value = _x; 
                        return x;
                    })
                    .Union(args.Through.OfType<Core.Parameter>()
                        .Where(x => x.Value is IEnumerable<DeZero.NET.Parameter>)
                        .SelectMany(x => x.Value as IEnumerable<DeZero.NET.Parameter>).Select(x =>
                        {
                            //GpuMemoryMonitor.Instance.LogMemoryUsage("if block begin 2");
                            if (x is DeZero.NET.Parameter pa &&
                                (pa?.Data?.Value is null || pa.Data.Value.Handle == IntPtr.Zero))
                            {
                                return null;
                            }

                            //GpuMemoryMonitor.Instance.LogMemoryUsage("if block end 2");
                            Variable parameter = x;
                            //GpuMemoryMonitor.Instance.LogMemoryUsage("if block 7");
                            parameter = parameter.copy();
                            //GpuMemoryMonitor.Instance.LogMemoryUsage("if block 8");
                            return new Core.Parameter(null, parameter);
                        })).Where(x => x is not null).ToList();
                //GpuMemoryMonitor.Instance.LogMemoryUsage("set Inputs end");



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
                        output.Dispose();
                    }
                }

                //GpuMemoryMonitor.Instance.LogMemoryUsage("set Outputs begin");
                this.Outputs = outputs.Select(x =>
                {
                    if (x is Variable pa && (pa?.Data?.Value is null || pa.Data.Value.Handle == IntPtr.Zero))
                    {
                        return null;
                    }

                    var newx = x.copy(holdReference: false);
                    newx.Grad.SetValueWithNoFireEvent(x.Grad.Value?.copy(holdReference: false));
                    return newx;
                }).Where(x => x is not null).ToList();
                //GpuMemoryMonitor.Instance.LogMemoryUsage("set Outputs end");

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
