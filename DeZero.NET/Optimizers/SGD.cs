namespace DeZero.NET.Optimizers
{
    public class SGD : Optimizer
    {
        public float lr { get; set; }

        public SGD(float lr = 0.01f) : base()
        {
            this.lr = lr;
        }

        public override void UpdateOne(Parameter param)
        {
            param.Data -= lr * param.Grad.Data;
        }
    }
}
