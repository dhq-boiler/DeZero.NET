using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class L2Regularization : Function
    {
        public override Variable[] Forward(Params args)
        {
            var parameters = args.Get<IEnumerable<Parameter>>(0);
            var lambda = args.Get<Variable>(1);
            var reg_loss = new NDarray(0d).ToVariable(this);

            foreach (var param in parameters.Where(x => x.Data?.Value is not null && x.Data.Value.Handle != IntPtr.Zero))
            {
                using var squared = (param.Data.Value * param.Data.Value).ToVariable(param);
                using var sum = squared.Data.Value.sum().ToVariable(squared);
                reg_loss += lambda * sum * 0.5;
            }

            return [reg_loss.Relay(this)];
        }

        public override Variable[] Backward(Params args)
        {
            var parameters = Inputs.ElementAt(0).Value as IEnumerable<Parameter>;
            var lambda = Inputs.ElementAt(1).Value as Variable;
            var grad = args.Get<Variable>(0);

            // Calculate gradients for each parameter
            foreach (var param in parameters.Where(x => x.Data?.Value is not null))
            {
                param.Grad.Value += lambda.Data.Value * param.Data.Value * grad.Data.Value;
            }

            // Return zero gradients for input arguments since they're not used in backward pass
            return new[] { new NDarray(0d).ToVariable(), new NDarray(0d).ToVariable() };
        }

        public static Variable[] Invoke(IEnumerable<Parameter> parameters, Variable lambda)
        {
            return new L2Regularization().Call(Params.New.SetPositionalArgs(parameters, lambda));
        }
    }
}
