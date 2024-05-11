using DeZero.NET.Core;

namespace DeZero.NET.Functions
{
    public class GetItemGrad : Function
    {
        public Slice[] Slices { get; set; }
        public Shape In_Shape { get; }

        public GetItemGrad(Slice[] slices, Shape in_shape)
        {
            Slices = slices;
            In_Shape = in_shape;
        }

        public override Variable[] Forward(Params args)
        {
            var gy = args.Get<Variable>("gy").Data;
            var gx = xp.zeros(In_Shape, dtype: gy.dtype);

            if (Gpu.Available && Gpu.Use)
            {
                xp.scatter_add(gx, Slices, gy);
            }
            else
            {
                xp.add_at(gx, Slices, gy);
            }

            return [gx.ToVariable()];
        }

        public override Variable[] Backward(Params args)
        {
            var ggx = args.Get<Variable>(0);
            return GetItem.Invoke(ggx, Slices);
        }
    }
}
