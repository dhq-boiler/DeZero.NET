using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class Clip : Function
    {
        private readonly double x_min;
        private readonly double x_max;

        public Clip(double x_min, double x_max)
        {
            // 入力値の検証を追加
            if (x_min > x_max)
            {
                throw new ArgumentException($"x_min ({x_min}) must be less than or equal to x_max ({x_max})");
            }

            this.x_min = x_min;
            this.x_max = x_max;
        }

        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>("x");
            if (x == null)
            {
                throw new ArgumentNullException(nameof(x), "Input variable cannot be null");
            }

            try
            {
                var y = xp.clip(x.Data.Value,
                               new NDarray(x_min),
                               new NDarray(x_max)).ToVariable(this);
                return [y];
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException("Error during forward pass of Clip function", ex);
            }
        }

        public override Variable[] Backward(Params args)
        {
            try
            {
                var gy = args.Get<Variable>(0);
                var x = Inputs.ElementAt(0).Variable;

                // マスクの計算を分割して、それぞれの条件を別々に計算
                var greater_equal = x.Data.Value >= x_min;
                var less_equal = x.Data.Value <= x_max;
                var mask = greater_equal * less_equal;

                var gx = gy * mask.ToVariable();
                return [gx];
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException("Error during backward pass of Clip function", ex);
            }
        }

        public static Variable[] Invoke(Variable x, double min, double max)
        {
            if (x == null)
            {
                throw new ArgumentNullException(nameof(x));
            }

            var clip = new Clip(min, max);
            return clip.Call(Params.New.SetKeywordArg(x));
        }

        // オーバーロードを追加して、float型でも使えるようにする
        public static Variable[] Invoke(Variable x, float min, float max)
        {
            return Invoke(x, (double)min, (double)max);
        }
    }
}