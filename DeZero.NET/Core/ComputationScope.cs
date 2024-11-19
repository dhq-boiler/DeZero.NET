using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeZero.NET.Core
{
    /// <summary>
    /// 計算スコープ管理用のヘルパークラス
    /// </summary>
    public class ComputationScope : IDisposable
    {
        private readonly List<Variable> _variables = new();

        public Variable Register(Variable var)
        {
            _variables.Add(var);
            return var;
        }

        public void Register(params Variable[] vars)
        {
            foreach (var var in vars)
            {
                _variables.Add(var);
            }
        }

        public void Dispose()
        {
            foreach (var var in _variables)
            {
                var?.Dispose();
            }
            _variables.Clear();
        }
    }
}
