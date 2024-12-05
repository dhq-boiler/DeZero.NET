using System.ComponentModel.DataAnnotations;
using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class Tensordot : Function
    {
        private readonly int[] axes1;
        private readonly int[] axes2;
        private Shape x1_shape;
        private Shape x2_shape;

        public Tensordot(int[] axes1, int[] axes2) : base()
        {
            this.axes1 = axes1;
            this.axes2 = axes2;
        }

        public override Variable[] Forward(Params args)
        {
            var x1 = args.Get<Variable>(0);
            var x2 = args.Get<Variable>(1);

            x1_shape = x1.Shape;
            x2_shape = x2.Shape;

            var y = xp.tensordot(x1.Data.Value, x2.Data.Value, [axes1, axes2]);
            return [y.Relay(this)];
        }

        public override Variable[] Backward(Params args)
        {
            var x1 = Inputs.ElementAt(0).Variable;
            var x2 = Inputs.ElementAt(1).Variable;
            var gy = args.Get<Variable>(0);

            var x1_ndim = x1.Data.Value.ndim;
            var x2_ndim = x2.Data.Value.ndim;

            //// Handle broadcasting in gradient computation
            //var x1_axes = Enumerable.Range(1, x1_ndim).Except(axes1).ToArray();
            //var x2_axes = Enumerable.Range(1, x2_ndim).Except(axes2).ToArray();

            //var deleteShape2_1 = x1.Shape.Dimensions.Except(gy.Shape.Dimensions).Select((x, i) =>  


            // Compute gradients
            var indices1 = GetSameSizeIndices(gy, x2, x1.Shape);
            var gx1 = xp.tensordot(gy.Data.Value, x2.Data.Value, [indices1.a, indices1.b]).reshape(x1.Shape);

            var indices2 = GetSameSizeIndices(x1, gy, x2.Shape);
            var gx2 = xp.tensordot(x1.Data.Value, gy.Data.Value, [indices2.a, indices2.b]).reshape(x2.Shape);

            //// aの軸を計算：[0]は最初の32用、[-2]は中間の14/112用、[-1]は最後の1/2用
            //var a_axes = new[] { 0, x1.ndim - 2, x1.ndim - 1 };

            //// bの軸を計算：[0]は最初の32用、[1]は中間の14/112用、[-2]は1/2用
            //var b_axes = new[] { 0, 1, gy.ndim - 2 };

            //// tensordotを実行し、必要な形状にreshape
            //var gx2 = xp.tensordot(x1.Data.Value, gy.Data.Value, [a_axes, b_axes])
            //    .reshape(new[] { gy.Shape.Dimensions.Last() }        // bの最後の次元
            //        .Concat(new[] { x1.Shape[1] }) // aの2番目の次元
            //        .Concat(new[] { 1, 1 })       // 固定の1,1
            //        .ToArray());

            return [gx1.ToVariable(), gx2.ToVariable()];
        }

        private static (int[] a, int[] b) GetSameSizeIndices(Variable x1, Variable gy, Shape target)
        {
            var a_shape = x1.Shape;
            var b_shape = gy.Shape;
            var a_shape_temp = a_shape.Dimensions.ToList();
            var b_shape_temp = b_shape.Dimensions.ToList();
            var sameSizes = a_shape_temp.Where(x =>
            {
                var ret = b_shape_temp.Contains(x);
                b_shape_temp.Remove(x);
                return ret;
            }).ToList();
            var sameSizeIndices_a = a_shape.Dimensions.Select((x, i) => new { Value = x, Index = i, P = "a" })
                .Where(x =>
                {
                    var ret = sameSizes.Contains(x.Value);
                    sameSizes.Remove(x.Value);
                    return ret;
                }).OrderByDescending(x => x.Value).ToArray();


            a_shape_temp = a_shape.Dimensions.ToList();
            b_shape_temp = b_shape.Dimensions.ToList();
            sameSizes = a_shape_temp.Where(x =>
            {
                var ret = b_shape_temp.Contains(x);
                b_shape_temp.Remove(x);
                return ret;
            }).ToList();
            var sameSizeIndices_b = b_shape.Dimensions.Select((x, i) => new {Value = x, Index = i, P = "b"})
                .Where(x =>
                {
                    var ret = sameSizes.Contains(x.Value);
                    sameSizes.Remove(x.Value);
                    return ret;
                }).OrderByDescending(x => x.Value).ToArray();

            if (true)
            {
                var targets = target.Dimensions
                    .GroupBy(x => x)
                    .Select(g => (Value: g.Key, Count: g.Count()))
                    .Where(x => x.Count >= 1);

                var sources = a_shape.Dimensions.Concat(b_shape.Dimensions)
                    .GroupBy(x => x)
                    .Select(g => (Value: g.Key, Count: g.Count()))
                    .Where(x => x.Count >= 1);

                var removals = sameSizeIndices_a.Concat(sameSizeIndices_b)
                    .GroupBy(x => x.Value)
                    .Select(g => (Value: g.Key, Count: g.Count()))
                    .Where(x => x.Count >= 1);

                foreach (var remove in removals)
                {
                    var t = sources.FirstOrDefault(x => x.Value == remove.Value);
                    if (t.Count >= remove.Count)
                    {
                        var p = t.Count - remove.Count;
                        t.Count = p;
                        var diff = t.Count - targets.FirstOrDefault(x => x.Value == remove.Value).Count;
                        if (diff < 0)
                        {
                            for (int i = 0; i < -diff; i++)
                            {
                                if (sameSizeIndices_a.Any(x => x.Value == remove.Value))
                                {
                                    sameSizeIndices_a =
                                        sameSizeIndices_a.Except([sameSizeIndices_a.First(x => x.Value == remove.Value)]).ToArray();
                                }
                                else if (sameSizeIndices_b.Any(x => x.Value == remove.Value))
                                {
                                    sameSizeIndices_b =
                                        sameSizeIndices_b.Except([sameSizeIndices_b.First(x => x.Value == remove.Value)]).ToArray();
                                }
                            }
                        }
                    }
                }
            }

            return (sameSizeIndices_a.Select(x => x.Index).ToArray(), sameSizeIndices_b.Select(x => x.Index).ToArray());
        }

        public static Variable[] Invoke(Variable x1, Variable x2, int[] axes1, int[] axes2)
        {
            return new Tensordot(axes1, axes2).Call(Params.New.SetPositionalArgs(x1, x2));
        }

        public static Variable[] Invoke(Variable x1, Variable x2, (int[], int[]) axes)
        {
            return new Tensordot(axes.Item1, axes.Item2).Call(Params.New.SetPositionalArgs(x1, x2));
        }

        public static Variable[] Invoke(Variable x1, Variable x2)
        {
            // Default case - equivalent to matrix multiplication
            var axes1 = new[] { x1.Data.Value.ndim - 1 };
            var axes2 = new[] { 0 };
            return new Tensordot(axes1, axes2).Call(Params.New.SetPositionalArgs(x1, x2));
        }
    }
}
