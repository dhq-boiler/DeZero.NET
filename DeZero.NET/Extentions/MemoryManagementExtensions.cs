using DeZero.NET.Core;
using DocumentFormat.OpenXml.VariantTypes;

namespace DeZero.NET.Extensions
{
    public static class MemoryManagementExtensions
    {
        private static void BasicDispose(this Variable variable)
        {
            try
            {
                if (variable?.Data?.Value is not null)
                {
                    variable.Data.Value.Dispose();
                    variable.Data.Value = null;
                }
                if (variable?.Grad?.Value?.Data?.Value is not null)
                {
                    variable.Grad.Value.Data.Value.Dispose();
                    variable.Grad.Value.Data.Value = null;
                }
            }
            finally
            {
                variable?.Dispose();
            }
        }

        public static void CleanupComputationalGraph(this Variable root)
        {
            if (root == null) return;

            var visitedVars = new HashSet<int>();
            var visitedFuncs = new HashSet<string>();
            var toProcess = new Queue<(Variable var, Function source, bool isInput)>();
            var disposalOrder = new List<Variable>();

            int processCount = 0;

            // ルートを処理キューに追加
            toProcess.Enqueue((root, null, false));
            disposalOrder.Add(root);
            processCount++;

            while (toProcess.Count > 0)
            {
                var (current, sourceFunc, isInput) = toProcess.Dequeue();
                if (current == null || visitedVars.Contains(current.Title)) continue;

                visitedVars.Add(current.Title);
                disposalOrder.Add(current);
                processCount++;

                // 1. Creatorチェーンの処理
                EnqueueFunctionChain(current.Creator, null, toProcess, visitedFuncs);

                // 2. CreatorListチェーンの処理
                if (current.CreatorList != null)
                {
                    foreach (var creator in current.CreatorList.Where(c => c != null))
                    {
                        EnqueueFunctionChain(creator, sourceFunc, toProcess, visitedFuncs);
                    }
                }

                // 3. Originsチェーンの処理
                if (current.Origins != null)
                {
                    foreach (var origin in current.Origins.Where(o => o != null))
                    {
                        EnqueueFunctionChain(origin, sourceFunc, toProcess, visitedFuncs);
                    }
                }
            }

            // 後処理：逆順で変数を解放
            for (int i = disposalOrder.Count - 1; i >= 0; i--)
            {
                var variable = disposalOrder[i];

                // 関連するFunctionの参照を解除
                if (variable.Creator is not null)
                {
                    variable.Creator.Inputs?.ToList().ForEach(x =>
                    {
                        if (x is Core.Parameter p)
                        {
                            p.Variable?.BasicDispose();
                        }
                    });
                    variable.Creator.Inputs = null;
                    variable.Creator.Outputs?.ToList().ForEach(x =>
                    {
                        x.BasicDispose();
                    });
                    if (variable.Creator is not null)
                    {
                        variable.Creator.Outputs = null;
                        variable.Creator = null;
                    }
                }

                if (variable.CreatorList is not null)
                {
                    foreach (var creator in variable.CreatorList)
                    {
                        if (creator is not null)
                        {
                            creator.Inputs?.ToList().ForEach(x =>
                            {
                                if (x is Core.Parameter p)
                                {
                                    p.Variable?.BasicDispose();
                                }
                            });
                            creator.Inputs = null;
                            creator.Outputs?.ToList().ForEach(x =>
                            {
                                x.BasicDispose();
                            });
                            creator.Outputs = null;
                        }
                    }
                    variable.CreatorList.Clear();
                    variable.CreatorList = null;
                }

                if (variable.Origins is not null)
                {
                    foreach (var origin in variable.Origins)
                    {
                        if (origin is not null)
                        {
                            origin.Inputs?.ToList().ForEach(x =>
                            {
                                if (x is Core.Parameter p)
                                {
                                    p.Variable?.BasicDispose();
                                }
                            });
                            origin.Inputs = null;
                            origin.Outputs?.ToList().ForEach(x =>
                            {
                                x.BasicDispose();
                            });
                            origin.Outputs = null;
                        }
                    }
                    variable.Origins = null;
                }

                variable.CopyGradToCloneSource = null;
                variable.BasicDispose();

            }

            GpuMemoryMonitor.Instance.LogMemoryUsage("Dispose");
        }

        private static void EnqueueFunctionChain(
            Function function,
            Function sourceFunc,
            Queue<(Variable var, Function source, bool isInput)> toProcess,
            HashSet<string> visitedFuncs)
        {
            if (function == null || visitedFuncs.Contains(function.Id)) return;
            visitedFuncs.Add(function.Id);

            // Inputsの連鎖を処理
            if (function.Inputs is not null)
            {
                foreach (var input in function.Inputs)
                {
                    if (input?.Variable is not null)
                    {
                        toProcess.Enqueue((input.Variable, function, true));
                    }
                }
            }

            // Outputsの連鎖を処理
            if (function.Outputs is not null)
            {
                foreach (var output in function.Outputs)
                {
                    if (output is not null && function.Id != sourceFunc?.Id)
                    {
                        toProcess.Enqueue((output, function, false));
                    }
                }
            }
        }
    }
}