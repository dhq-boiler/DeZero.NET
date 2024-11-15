using System;

namespace DeZero.NET.Extensions;

public static class MemoryManagementExtensions
{
    // NDarrayやVariableの配列を安全にDisposeする拡張メソッド
    public static void SafeDisposeAll(this IEnumerable<object> objects)
    {
        if (objects == null) return;

        foreach (var obj in objects.Where(x => x != null))
        {
            try
            {
                switch (obj)
                {
                    case Variable v:
                        v.Dispose();
                        break;
                    case NDarray n:
                        n.Dispose();
                        break;
                    case IDisposable d:
                        d.Dispose();
                        break;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error disposing object: {ex.Message}");
            }
        }
    }

    // 計算グラフをクリーンアップするヘルパーメソッド
    //public static void CleanupComputationalGraph(this Variable root)
    //{
    //    if (root == null) return;

    //    var visited = new HashSet<Variable>();
    //    var toDispose = new Stack<Variable>();

    //    void Visit(Variable node)
    //    {
    //        if (node == null || visited.Contains(node)) return;
    //        visited.Add(node);

    //        if (node.Creator != null)
    //        {
    //            foreach (var input in node.Creator.Inputs)
    //            {
    //                Visit(input.Variable);
    //            }
    //        }
    //        toDispose.Push(node);
    //    }

    //    Visit(root);

    //    while (toDispose.Count > 0)
    //    {
    //        var node = toDispose.Pop();
    //        if (node.Creator != null)
    //        {
    //            node.Creator = null;
    //        }
    //        if (node.Grad != null)
    //        {
    //            node.Grad.Value?.Dispose();
    //            node.Grad.Value = null;
    //        }
    //        node?.Dispose();
    //        node = null;
    //    }
    //}

    public static void CleanupComputationalGraph(this Variable root)
    {
        if (root == null) return;

        var visited = new HashSet<Variable>();
        var toDispose = new Stack<Variable>();

        void Visit(Variable node)
        {
            if (node == null || visited.Contains(node)) return;
            visited.Add(node);

            if (node.Creator != null)
            {
                foreach (var input in node.Creator.Inputs)
                {
                    Visit(input.Variable);
                }
            }
            toDispose.Push(node);
        }

        Visit(root);

        while (toDispose.Count > 0)
        {
            var node = toDispose.Pop();
            if (node.Creator != null)
            {
                node.Creator = null;
            }
            if (node.Grad != null)
            {
                node.Grad.Value?.Dispose();
                node.Grad.Value = null;
            }
        }
    }

    // バッチ処理用の一時変数を管理するためのコンテキストマネージャ
    public class BatchScope : IDisposable
    {
        private readonly List<IDisposable> _temporaryVariables = new();

        public void RegisterForDisposal(IDisposable disposable)
        {
            _temporaryVariables.Add(disposable);
        }

        public void Dispose()
        {
            foreach (var disposable in _temporaryVariables)
            {
                try
                {
                    disposable?.Dispose();
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error disposing temporary variable: {ex.Message}");
                }
            }
            _temporaryVariables.Clear();
        }
    }
}