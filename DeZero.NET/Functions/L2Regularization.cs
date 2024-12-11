using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class L2Regularization : Function
    {
        public override Variable[] Forward(Params args)
        {
            var parameters = args.Get<IEnumerable<Parameter>>(0);
            var hyperParameter = args.Get<Variable>(1);
            var reg_loss = new NDarray(0d).ToVariable(this);
            foreach (var param in parameters)
            {
                using var param_param = (param.Data.Value * param.Data.Value).ToVariable(param);
                using var param_param_sum = param_param.Data.Value.sum().ToVariable(param_param);
                using var a = hyperParameter * param_param_sum;
                using var b = a * 0.5;
                reg_loss +=  b;
            }
            return [reg_loss.Relay(this)];
        }

        public override Variable[] Backward(Params args)
        {
            return args.Through.Where(x => x.Value is not null).Select(p => p.NDarray.copy().ToVariable()).ToArray();
        }

        public static Variable[] Invoke(IEnumerable<Parameter> parameters, Variable lambda)
        {
            return new L2Regularization().Call(Params.New.SetPositionalArgs(parameters, lambda));
        }
    }
}
