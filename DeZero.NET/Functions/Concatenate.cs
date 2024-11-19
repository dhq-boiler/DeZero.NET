using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class Concatenate : Function
    {
        public int Axis { get; set; }
        public Shape[] Shapes { get; private set; }

        public Concatenate(int axis = 0)
        {
            this.Axis = axis;
        }

        public override Variable[] Forward(Params args)
        {
            var xs = args.Through.Select(p => p.NDarray).ToArray();
            this.Shapes = xs.Select(x => x.shape).ToArray();
            return [xp.concatenate(xs, this.Axis).ToVariable(this)];
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Get<Variable>(0);
            var gxs = new List<Variable>();
            int prev_end = 0;
            foreach (var shape in this.Shapes)
            {
                var start = prev_end;
                var end = start + shape[this.Axis];
                if (this.Axis == 0)
                {
                    gxs.Add(gy.Data.Value[new NET.Slice(start, end)].ToVariable());
                }
                else
                {
                    var indices = new NET.Slice() * gy.ndim;
                    indices[this.Axis] = new NET.Slice(start, end);
                    gxs.Add(gy.Data.Value[indices].ToVariable());
                }
                prev_end = end;
            }
            return gxs.ToArray();
        }

        public static Variable[] Invoke(Variable x, int axis = 1)
        {
            return new Concatenate(axis).Call(Params.New.SetPositionalArgs(x));
        }

        public static Variable[] Invoke(Variable x1, Variable x2, int axis = 1)
        {
            return new Concatenate(axis).Call(Params.New.SetPositionalArgs(x1, x2));
        }

        public static Variable[] Invoke(Variable[] xs, int axis = 1)
        {
            var paramz = Params.New;
            xs.ToList().ForEach(x => paramz = paramz.SetPositionalArgs(x));
            return new Concatenate(axis).Call(paramz);
        }
    }
}
