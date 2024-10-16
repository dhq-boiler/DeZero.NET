using DeZero.NET;

namespace MovieFileDataLoaderSampleWorker
{
    public static class TensorSlicingExtensions
    {
        public static Variable GetLastTimeStep(this Variable tensor)
        {
            if (tensor.Data.Value.shape.Dimensions.Count() < 3)
            {
                throw new ArgumentException("Tensor must have at least 3 dimensions (batch, time, features)");
            }

            int batchSize = tensor.Shape[0];
            int timeSteps = tensor.Shape[1];
            int features = tensor.Shape[2];

            // Reshape to (batch * time, features)
            var reshaped = DeZero.NET.Functions.Reshape.Invoke(tensor, new Shape(batchSize * timeSteps, features))[0];

            // Get the last time step for each batch
            var indices = xp.array(Enumerable.Range(0, batchSize).Select(i => (i + 1) * timeSteps - 1).ToArray());
            var lastTimeStep = DeZero.NET.Functions.GetItem.Invoke(reshaped, indices)[0];

            // Reshape back to (batch, features)
            return DeZero.NET.Functions.Reshape.Invoke(lastTimeStep, new Shape(batchSize, features))[0];
        }
    }
}
