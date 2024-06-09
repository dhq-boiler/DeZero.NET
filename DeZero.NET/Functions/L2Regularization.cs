using DeZero.NET.Core;

namespace DeZero.NET.Functions
{
    public class L2Regularization : Function
    {
        public override Variable[] Forward(Params args)
        {
            var parameters = args.Get<IEnumerable<Parameter>>(0);
            var hyperParameter = args.Get<Variable>(1);
            var reg_loss = new NDarray(0d).ToVariable(this);
            foreach (var param in parameters.Skip(1))
            {
                reg_loss += hyperParameter * (param.Data.Value * param.Data.Value).sum() * 0.5;
            }
            return [reg_loss];
        }

        public override Variable[] Backward(Params args)
        {
            return args.Through.Select(p => p.NDarray.copy().ToVariable()).ToArray();
        }

        public static Variable[] Invoke(IEnumerable<Parameter> parameters, Variable lambda)
        {
            return new L2Regularization().Call(Params.New.SetPositionalArgs(parameters, lambda));
        }
    }
}
