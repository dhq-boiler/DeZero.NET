using DeZero.NET.Log;

namespace DeZero.NET.Core
{
    public static class DimensionHelper
    {
        public static Variable EnsureShape(Variable x, int expectedDim, ILogger logger)
        {
            try
            {
                if (x.ndim != expectedDim)
                {
                    logger.LogWarning($"Unexpected dimensions: {x.ndim}, expected: {expectedDim}");
                    logger.LogDebug($"Current shape: {string.Join(", ", x.Shape.Dimensions)}");

                    // CNNからの出力の場合（4次元から2次元へ）
                    if (x.ndim == 4 && expectedDim == 2)
                    {
                        var batchSize = x.Shape[0];
                        var totalFeatures = x.Shape[1] * x.Shape[2] * x.Shape[3];
                        return DeZero.NET.Functions.Reshape.Invoke(x,
                            new Shape(batchSize, totalFeatures))[0];
                    }

                    // その他の場合
                    throw new InvalidOperationException(
                        $"Cannot automatically reshape from {x.ndim}D to {expectedDim}D");
                }
                return x;
            }
            catch (Exception ex)
            {
                logger.LogError($"Shape conversion error: {ex.Message}");
                throw;
            }
        }

        public static void ValidateMatrixDimensions(Variable x, Variable weights, string operationName, ILogger logger)
        {
            var xShape = string.Join(", ", x.Shape.Dimensions);
            var wShape = string.Join(", ", weights.Shape.Dimensions);
            logger.LogDebug($"{operationName} - Input shapes: x={xShape}, weights={wShape}");

            if (x.Shape.Dimensions.Last() != weights.Shape[0])
            {
                throw new InvalidOperationException(
                    $"Matrix dimension mismatch in {operationName}: {x.Shape.Dimensions.Last()} != {weights.Shape[0]}");
            }
        }
    }
}
