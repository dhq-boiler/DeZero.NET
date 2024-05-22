using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace DeZero.NET.Core
{
    public class Parameter
    {
        public string Name { get; set; }
        public object Value { get; set; }
        public Variable Variable => Value as Variable ?? (Value as NDarray).ToVariable();
        public NDarray NDarray => Value as NDarray ?? (Value as Variable).Data.Value;

        [DebuggerStepThrough]
        public Parameter(string name, object value)
        {
            Name = name;
            Value = value;
        }

        public override bool Equals(object? obj)
        {
            return obj is Parameter parameter &&
                   Name == parameter.Name &&
                   Value.Equals(parameter.Value);
        }

        public override int GetHashCode()
        {
            return Name.GetHashCode();
        }
    }

    public class Params
    {
        protected readonly List<Parameter> _positional_args = new();
        protected readonly Dictionary<string, Parameter> _keyword_args = new();

        public Dictionary<string, Parameter> KeywordArgs => _keyword_args;

        public static readonly Params Empty = new();

        public static Params New => new();

        /// <summary>
        /// 引数 args を基に新しい Params インスタンスを生成します。
        /// </summary>
        /// <param name="args"></param>
        /// <returns></returns>
        //[DebuggerStepThrough]
        public static Params Base(Params args)
        {
            var pc = new Params();
            pc._positional_args.AddRange(args._positional_args);
            foreach (var item in args._keyword_args)
            {
                pc._keyword_args[item.Key] = item.Value;
            }
            return pc;
        }

        [DebuggerStepThrough]
        public Params()
        {
        }

        public virtual T Get<T>(string key)
        {
            var ret = InternalGet<T>(key);
            return (T)(ret is NDarray arr ? arr.ToVariable() : ret);
        }

        public virtual T Get<T>(string key, T defaultValue)
        {
            return Get<T>(key) ?? defaultValue;
        }

        public virtual T Get<T>(int index) where T : class
        {
            var ret = InnerThrough().ElementAt(index).Value;
            if (typeof(T).Name == "Variable")
            {
                Variable a = ret switch
                {
                    NDarray arr => arr.ToVariable(),
                    NDarray[] arrs => arrs[0].ToVariable(),
                    Variable v => v,
                    Variable[] vars => vars[0],
                    Property<NDarray> p => p.Value.ToVariable(),
                    Property<Variable> p => p.Value,
                    _ => (Variable)ret
                };
                return a as T;
            }
            else
            {
                return (T)ret;
            }
        }

        public virtual T Get<T>(int index, T defaultValue) where T : class
        {
            if (_positional_args.Count > index && index >= 0)
            {
                return Get<T>(index);
            }
            else
            {
                return defaultValue;
            }
        }

        [DebuggerStepThrough]
        private object InternalGet<T>(string key)
        {
            foreach (var item in _positional_args)
            {
                if (item is not Core.Parameter p) continue;

                if (p.Name == key)
                {
                    return p.Value;
                }
            }

            foreach (var pair in _keyword_args)
            {
                if (pair.Value.Value is Params p)
                {
                    var ret = p.InternalGet<T?>(key);
                    if (ret is not null)
                    {
                        return ret;
                    }
                }
            }

            if (_keyword_args.ContainsKey(key))
            {
                var ret = _keyword_args[key].Value;
                if (ret is Core.Parameter p)
                {
                    return p.Variable;
                }
                else if (ret is NDarray arr)
                {
                    return arr.ToVariable();
                }

                return (T)ret;
            }

            return null;
        }

        #region SetKeywordArg

        //[DebuggerStepThrough]
        public Params SetKeywordArg<T1>(T1 value1, [CallerArgumentExpression(nameof(value1))] string? keyword1 = null)
        {
            _keyword_args[keyword1] = new Parameter(keyword1, value1);
            return this;
        }

        [DebuggerStepThrough]
        public Params SetKeywordArg<T1, T2>(T1 value1, T2 value2,
            [CallerArgumentExpression(nameof(value1))] string? keyword1 = null,
            [CallerArgumentExpression(nameof(value2))] string? keyword2 = null)
        {
            _keyword_args[keyword1] = new Parameter(keyword1, value1);
            _keyword_args[keyword2] = new Parameter(keyword2, value2);
            return this;
        }

        [DebuggerStepThrough]
        public Params SetKeywordArg<T1, T2, T3>(T1 value1, T2 value2, T3 value3,
            [CallerArgumentExpression(nameof(value1))] string? keyword1 = null,
            [CallerArgumentExpression(nameof(value2))] string? keyword2 = null,
            [CallerArgumentExpression(nameof(value3))] string? keyword3 = null)
        {
            _keyword_args[keyword1] = new Parameter(keyword1, value1);
            _keyword_args[keyword2] = new Parameter(keyword2, value2);
            _keyword_args[keyword3] = new Parameter(keyword3, value3);
            return this;
        }

        [DebuggerStepThrough]
        public Params SetKeywordArg<T1, T2, T3, T4>(T1 value1, T2 value2, T3 value3, T4 value4,
            [CallerArgumentExpression(nameof(value1))] string? keyword1 = null,
            [CallerArgumentExpression(nameof(value2))] string? keyword2 = null,
            [CallerArgumentExpression(nameof(value3))] string? keyword3 = null,
            [CallerArgumentExpression(nameof(value4))] string? keyword4 = null)
        {
            _keyword_args[keyword1] = new Parameter(keyword1, value1);
            _keyword_args[keyword2] = new Parameter(keyword2, value2);
            _keyword_args[keyword3] = new Parameter(keyword3, value3);
            _keyword_args[keyword4] = new Parameter(keyword4, value4);
            return this;
        }

        [DebuggerStepThrough]
        public Params SetKeywordArg<T1, T2, T3, T4, T5>(T1 value1, T2 value2, T3 value3, T4 value4, T5 value5,
            [CallerArgumentExpression(nameof(value1))] string? keyword1 = null,
            [CallerArgumentExpression(nameof(value2))] string? keyword2 = null,
            [CallerArgumentExpression(nameof(value3))] string? keyword3 = null,
            [CallerArgumentExpression(nameof(value4))] string? keyword4 = null,
            [CallerArgumentExpression(nameof(value5))] string? keyword5 = null)
        {
            _keyword_args[keyword1] = new Parameter(keyword1, value1);
            _keyword_args[keyword2] = new Parameter(keyword2, value2);
            _keyword_args[keyword3] = new Parameter(keyword3, value3);
            _keyword_args[keyword4] = new Parameter(keyword4, value4);
            _keyword_args[keyword5] = new Parameter(keyword5, value5);
            return this;
        }

        [DebuggerStepThrough]
        public Params SetKeywordArg<T1, T2, T3, T4, T5, T6>(T1 value1, T2 value2, T3 value3, T4 value4, T5 value5, T6 value6,
            [CallerArgumentExpression(nameof(value1))] string? keyword1 = null,
            [CallerArgumentExpression(nameof(value2))] string? keyword2 = null,
            [CallerArgumentExpression(nameof(value3))] string? keyword3 = null,
            [CallerArgumentExpression(nameof(value4))] string? keyword4 = null,
            [CallerArgumentExpression(nameof(value5))] string? keyword5 = null,
            [CallerArgumentExpression(nameof(value6))] string? keyword6 = null)
        {
            _keyword_args[keyword1] = new Parameter(keyword1, value1);
            _keyword_args[keyword2] = new Parameter(keyword2, value2);
            _keyword_args[keyword3] = new Parameter(keyword3, value3);
            _keyword_args[keyword4] = new Parameter(keyword4, value4);
            _keyword_args[keyword5] = new Parameter(keyword5, value5);
            _keyword_args[keyword6] = new Parameter(keyword6, value6);
            return this;
        }

        [DebuggerStepThrough]
        public Params SetKeywordArg<T1, T2, T3, T4, T5, T6, T7>(T1 value1, T2 value2, T3 value3, T4 value4, T5 value5, T6 value6, T7 value7,
            [CallerArgumentExpression(nameof(value1))] string? keyword1 = null,
            [CallerArgumentExpression(nameof(value2))] string? keyword2 = null,
            [CallerArgumentExpression(nameof(value3))] string? keyword3 = null,
            [CallerArgumentExpression(nameof(value4))] string? keyword4 = null,
            [CallerArgumentExpression(nameof(value5))] string? keyword5 = null,
            [CallerArgumentExpression(nameof(value6))] string? keyword6 = null,
            [CallerArgumentExpression(nameof(value7))] string? keyword7 = null)
        {
            _keyword_args[keyword1] = new Parameter(keyword1, value1);
            _keyword_args[keyword2] = new Parameter(keyword2, value2);
            _keyword_args[keyword3] = new Parameter(keyword3, value3);
            _keyword_args[keyword4] = new Parameter(keyword4, value4);
            _keyword_args[keyword5] = new Parameter(keyword5, value5);
            _keyword_args[keyword6] = new Parameter(keyword6, value6);
            _keyword_args[keyword7] = new Parameter(keyword7, value7);
            return this;
        }

        [DebuggerStepThrough]
        public Params SetKeywordArg<T1, T2, T3, T4, T5, T6, T7, T8>(T1 value1, T2 value2, T3 value3, T4 value4, T5 value5, T6 value6, T7 value7, T8 value8,
            [CallerArgumentExpression(nameof(value1))] string? keyword1 = null,
            [CallerArgumentExpression(nameof(value2))] string? keyword2 = null,
            [CallerArgumentExpression(nameof(value3))] string? keyword3 = null,
            [CallerArgumentExpression(nameof(value4))] string? keyword4 = null,
            [CallerArgumentExpression(nameof(value5))] string? keyword5 = null,
            [CallerArgumentExpression(nameof(value6))] string? keyword6 = null,
            [CallerArgumentExpression(nameof(value7))] string? keyword7 = null,
            [CallerArgumentExpression(nameof(value8))] string? keyword8 = null)
        {
            _keyword_args[keyword1] = new Parameter(keyword1, value1);
            _keyword_args[keyword2] = new Parameter(keyword2, value2);
            _keyword_args[keyword3] = new Parameter(keyword3, value3);
            _keyword_args[keyword4] = new Parameter(keyword4, value4);
            _keyword_args[keyword5] = new Parameter(keyword5, value5);
            _keyword_args[keyword6] = new Parameter(keyword6, value6);
            _keyword_args[keyword7] = new Parameter(keyword7, value7);
            _keyword_args[keyword8] = new Parameter(keyword8, value8);
            return this;
        }

        #endregion

        #region SetPositionalArgs

        [DebuggerStepThrough]
        private void RemoveExistsFromPositionalArgs(string? arg1Name, object obj)
        {
            for (int i = 0; i < _positional_args.Count; i++)
            {
                if (_positional_args[i].Name == arg1Name && _positional_args[i].Value.Equals(obj))
                {
                    _positional_args.RemoveAt(i);
                    break;
                }
            }
        }

        [DebuggerStepThrough]
        public Params SetPositionalArgs(object arg1, [CallerArgumentExpression(nameof(arg1))] string? arg1Name = null)
        {
            OverwritePositionalArgs(arg1Name, arg1);
            return this;
        }

        [DebuggerStepThrough]
        public Params OverwritePositionalArgs(string? arg1Name, object arg1)
        {
            var p = new Parameter(arg1Name, arg1);

            if (_positional_args.Contains(p))
            {
                //_positional_argsの中に同じParameterが存在する場合は上書きする
                for (int i = 0; i < _positional_args.Count; i++)
                {
                    if (_positional_args[i] == p)
                    {
                        _positional_args[i] = p;
                    }
                }
            }
            else
            {
                this._positional_args.Add(p);
            }

            return this;
        }

        [DebuggerStepThrough]
        public Params SetPositionalArgs<T1>(T1[] args)
        {
            this._positional_args.AddRange(args.Cast<object>().Select(x => new Parameter("@", x)));
            return this;
        }

        [DebuggerStepThrough]
        public Params SetPositionalArgs<T1, T2>(T1 arg1, T2 arg2, [CallerArgumentExpression(nameof(arg1))] string? arg1Name = null, [CallerArgumentExpression(nameof(arg2))] string? arg2Name = null)
        {
            OverwritePositionalArgs(arg1Name, arg1);
            OverwritePositionalArgs(arg2Name, arg2);
            return this;
        }

        [DebuggerStepThrough]
        public Params SetPositionalArgs<T1, T2, T3>(T1 arg1, T2 arg2, T3 arg3,
            [CallerArgumentExpression(nameof(arg1))] string? arg1Name = null,
            [CallerArgumentExpression(nameof(arg2))] string? arg2Name = null,
            [CallerArgumentExpression(nameof(arg3))] string? arg3Name = null)
        {
            OverwritePositionalArgs(arg1Name, arg1);
            OverwritePositionalArgs(arg2Name, arg2);
            OverwritePositionalArgs(arg3Name, arg3);
            return this;
        }

        [DebuggerStepThrough]
        public Params SetPositionalArgs<T1, T2, T3, T4>(T1 arg1, T2 arg2, T3 arg3, T4 arg4,
            [CallerArgumentExpression(nameof(arg1))] string? arg1Name = null,
            [CallerArgumentExpression(nameof(arg2))] string? arg2Name = null,
            [CallerArgumentExpression(nameof(arg3))] string? arg3Name = null,
            [CallerArgumentExpression(nameof(arg4))] string? arg4Name = null)
        {
            OverwritePositionalArgs(arg1Name, arg1);
            OverwritePositionalArgs(arg2Name, arg2);
            OverwritePositionalArgs(arg3Name, arg3);
            OverwritePositionalArgs(arg4Name, arg4);
            return this;
        }

        [DebuggerStepThrough]
        public Params SetPositionalArgs<T1, T2, T3, T4, T5>(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5,
            [CallerArgumentExpression(nameof(arg1))] string? arg1Name = null,
            [CallerArgumentExpression(nameof(arg2))] string? arg2Name = null,
            [CallerArgumentExpression(nameof(arg3))] string? arg3Name = null,
            [CallerArgumentExpression(nameof(arg4))] string? arg4Name = null,
            [CallerArgumentExpression(nameof(arg5))] string? arg5Name = null)
        {
            OverwritePositionalArgs(arg1Name, arg1);
            OverwritePositionalArgs(arg2Name, arg2);
            OverwritePositionalArgs(arg3Name, arg3);
            OverwritePositionalArgs(arg4Name, arg4);
            OverwritePositionalArgs(arg5Name, arg5);
            return this;
        }

        [DebuggerStepThrough]
        public Params SetPositionalArgs<T1, T2, T3, T4, T5, T6>(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6,
            [CallerArgumentExpression(nameof(arg1))] string? arg1Name = null,
            [CallerArgumentExpression(nameof(arg2))] string? arg2Name = null,
            [CallerArgumentExpression(nameof(arg3))] string? arg3Name = null,
            [CallerArgumentExpression(nameof(arg4))] string? arg4Name = null,
            [CallerArgumentExpression(nameof(arg5))] string? arg5Name = null,
            [CallerArgumentExpression(nameof(arg6))] string? arg6Name = null)
        {
            OverwritePositionalArgs(arg1Name, arg1);
            OverwritePositionalArgs(arg2Name, arg2);
            OverwritePositionalArgs(arg3Name, arg3);
            OverwritePositionalArgs(arg4Name, arg4);
            OverwritePositionalArgs(arg5Name, arg5);
            OverwritePositionalArgs(arg6Name, arg6);
            return this;
        }

        [DebuggerStepThrough]
        public Params SetPositionalArgs<T1, T2, T3, T4, T5, T6, T7>(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7,
            [CallerArgumentExpression(nameof(arg1))] string? arg1Name = null,
            [CallerArgumentExpression(nameof(arg2))] string? arg2Name = null,
            [CallerArgumentExpression(nameof(arg3))] string? arg3Name = null,
            [CallerArgumentExpression(nameof(arg4))] string? arg4Name = null,
            [CallerArgumentExpression(nameof(arg5))] string? arg5Name = null,
            [CallerArgumentExpression(nameof(arg6))] string? arg6Name = null,
            [CallerArgumentExpression(nameof(arg7))] string? arg7Name = null)
        {
            OverwritePositionalArgs(arg1Name, arg1);
            OverwritePositionalArgs(arg2Name, arg2);
            OverwritePositionalArgs(arg3Name, arg3);
            OverwritePositionalArgs(arg4Name, arg4);
            OverwritePositionalArgs(arg5Name, arg5);
            OverwritePositionalArgs(arg6Name, arg6);
            OverwritePositionalArgs(arg7Name, arg7);
            return this;
        }
        
        [DebuggerStepThrough]
        public Params SetPositionalArgs<T1, T2, T3, T4, T5, T6, T7, T8>(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8,
            [CallerArgumentExpression(nameof(arg1))] string? arg1Name = null,
            [CallerArgumentExpression(nameof(arg2))] string? arg2Name = null,
            [CallerArgumentExpression(nameof(arg3))] string? arg3Name = null,
            [CallerArgumentExpression(nameof(arg4))] string? arg4Name = null,
            [CallerArgumentExpression(nameof(arg5))] string? arg5Name = null,
            [CallerArgumentExpression(nameof(arg6))] string? arg6Name = null,
            [CallerArgumentExpression(nameof(arg7))] string? arg7Name = null,
            [CallerArgumentExpression(nameof(arg8))] string? arg8Name = null)
        {
            OverwritePositionalArgs(arg1Name, arg1);
            OverwritePositionalArgs(arg2Name, arg2);
            OverwritePositionalArgs(arg3Name, arg3);
            OverwritePositionalArgs(arg4Name, arg4);
            OverwritePositionalArgs(arg5Name, arg5);
            OverwritePositionalArgs(arg6Name, arg6);
            OverwritePositionalArgs(arg7Name, arg7);
            OverwritePositionalArgs(arg8Name, arg8);
            return this;
        }

        #endregion

        public virtual Parameter[] Through => InnerThrough().ToArray();

        [DebuggerStepThrough]
        private IEnumerable<Parameter> InnerThrough()
        {
            foreach (var item in _positional_args.Distinct())
            {
                if (item.Value is Params p)
                {
                    foreach (var v in p.Through)
                    {
                        yield return v;
                    }
                }
                else
                {
                    yield return item;
                }
            }

            foreach (var item in _keyword_args.Values)
            {
                if (item.Value is Params p)
                {
                    foreach (var v in p.Through)
                    {
                        yield return v;
                    }
                }
                else
                {
                    yield return item;
                }
            }
        }

        public Params AddRange(IEnumerable<Variable> enumerable)
        {
            foreach (var item in enumerable)
            {
                _positional_args.Add(new Parameter(item.Name.Value, item));
            }
            return this;
        }
    }
}
