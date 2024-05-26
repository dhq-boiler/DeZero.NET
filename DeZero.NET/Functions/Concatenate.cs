using DeZero.NET.Core;

namespace DeZero.NET.Functions
{
    public class Concatenate : Function
    {
        public int Axis { get; set; }
        public Shape[] Shapes { get; private set; }

        public Concatenate(int axis = 1)
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
                    gxs.Add(gy.Data.Value[new Slice(start, end)].ToVariable());
                }
                else
                {
                    var indices = new Slice() * gy.ndim;
                    indices[this.Axis] = new Slice(start, end);
                    gxs.Add(gy.Data.Value[indices].ToVariable());
                }
                prev_end = end;
            }
            return gxs.ToArray();
        }

        public static Variable Invoke(Variable x, int axis = 1)
        {
            return new Concatenate(axis).Call(Params.New.SetPositionalArgs(x))[0];
        }
    }
}
