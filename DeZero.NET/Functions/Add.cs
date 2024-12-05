using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class Add : Function
    {
        private bool disposed = false;
        public static Func<Params, NDarray[]> F => x => [(x.Get<Variable>(0).Data.Value + x.Get<Variable>(1).Data.Value)];
        public Shape X0_Shape { get; set; }
        public Shape X1_Shape { get; set; }

        public Add()
        { }

        public Add(Func<Params, Variable[]> f)
            : base(f)
        { }

        public override Variable[] Forward(Params args)
        {
            var xs = args.Through;

            // 既存のShapeをクリーンアップ
            CleanupShapes();

            // 新しいShapeを保存
            X0_Shape = new Shape(xs[0].Variable.Shape.Dimensions);
            X1_Shape = new Shape(xs[1].Variable.Shape.Dimensions);

            using var result = F(Params.New.SetPositionalArgs(xs[0].Value, xs[1].Value))[0];
            return new[] { result.copy().Relay(this, [..xs.Select(x => x.Variable)]) };
        }

        public override Variable[] Backward(Params args)
        {
            using (var scope = new ComputationScope())
            {
                var gys = args.Through;
                var gx0 = gys[0].Variable;
                var gx1 = gys[0].Variable;

                if (!ShapesAreEqual(X0_Shape, X1_Shape))
                {
                    gx0 = SumTo.Invoke(gx0, X0_Shape)[0];
                    gx1 = SumTo.Invoke(gx1, X1_Shape)[0];
                }

                return new[] { gx0, gx1 };
            }
        }

        private bool ShapesAreEqual(Shape s1, Shape s2)
        {
            if (s1 == null || s2 == null) return false;
            return s1.Dimensions.SequenceEqual(s2.Dimensions);
        }

        private void CleanupShapes()
        {
            X0_Shape?.Dispose();
            X1_Shape?.Dispose();
            X0_Shape = null;
            X1_Shape = null;
        }

        public void Dispose()
        {
            if (!disposed)
            {
                CleanupShapes();
                disposed = true;
            }
            GC.SuppressFinalize(this);
        }

        public static (Variable[], Add) Invoke(Variable x0, Variable x1)
        {
            var op = new Add();
            return (op.Call(Params.New.SetPositionalArgs(x0, x1)), op);
        }

        public static Variable[] Invoke(Add op, Variable x0, Variable x1)
        {
            return op.Call(Params.New.SetPositionalArgs(x0, x1));
        }
    }
}
