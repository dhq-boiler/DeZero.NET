using DeZero.NET.Layers;

namespace DeZero.NET.Optimizers.HookFunctions
{
    public class FreezeParam : HookFunction
    {
        public List<Parameter> FreezeParams { get; set; }

        public FreezeParam(Layer[] layers)
        {
            FreezeParams = new List<Parameter>();
            foreach (var layer in layers)
            {
                foreach (var p in layer.Params())
                {
                    FreezeParams.Add(p);
                }
            }
        }

        public override void Call(Parameter[] @params)
        {
            foreach (var p in FreezeParams)
            {
                p.Grad.Value = null;
            }
        }
    }
}
