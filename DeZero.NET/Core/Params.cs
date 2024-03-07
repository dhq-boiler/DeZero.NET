using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace DeZero.NET.Core
{
    public abstract class Params
    {
        private readonly Dictionary<string, object> _dictionary = new();
        private readonly List<Variable> _list = new();
        protected readonly List<object> _objlist = new();

        public Params()
        {
        }

        public virtual T Get<T>(string key)
        {
            return (T)InternalGet<T>(key);
        }

        public virtual T Get<T>(string key, T defaultValue)
        {
            return (T)InternalGet<T>(key) ?? defaultValue;
        }

        public virtual T Get<T>(int index)
        {
            return (T)_objlist[index];
        }

        public virtual T Get<T>(int index, T defaultValue)
        {
            if (_objlist.Count > index && index >= 0)
            {
                return Get<T>(index);
            }
            else
            {
                return defaultValue;
            }
        }

        private object InternalGet<T>(string key)
        {
            foreach (var pair in _dictionary)
            {
                if (pair.Value is Params p)
                {
                    var ret = p.InternalGet<T?>(key);
                    if (ret is not null)
                    {
                        return ret;
                    }
                }
            }

            if (_dictionary.ContainsKey(key))
            {
                return (T)_dictionary[key];
            }

            return null;
        }

        public Params Set<T>(string key, T value)
        {
            _dictionary[key] = value;
            if (typeof(T) == typeof(Variable))
            {
                _list.Add(value as Variable);
            }

            _objlist.Add(value);
            return this;
        }

        public Variable[] Through() => InnerThrough().ToArray();

        private IEnumerable<Variable> InnerThrough()
        {
            if (this is OrderedParams)
            {
                foreach (var item in _objlist)
                {
                    yield return item as Variable;
                }
                yield break;
            }

            foreach (var item in _list)
            {
                yield return item;
            }

            foreach (var item in _dictionary.Values.OfType<Params>().SelectMany(x => x.InnerThrough()))
            {
                yield return item;
            }
        }
    }

    public class Params<T1> : Params
    {
        public static Params<T1> args<T1>(T1 arg1, [CallerArgumentExpression(nameof(arg1))] string? arg1Name = null)
        {
            var pc = new Params<T1>();
            pc.Set(arg1Name, arg1);
            return pc;
        }

        public Params<T1> SetParams<T2>(Params kwargs)
        {
            Set<Params>(nameof(kwargs), kwargs);
            return this;
        }
    }

    public class Params<T1, T2> : Params
    {
        public static Params<T1, T2> args<T1, T2>(T1 arg1, T2 arg2, [CallerArgumentExpression(nameof(arg1))] string? arg1Name = null, [CallerArgumentExpression(nameof(arg2))] string? arg2Name = null)
        {
            var pc = new Params<T1, T2>();
            pc.Set(arg1Name, arg1);
            pc.Set(arg2Name, arg2);
            return pc;
        }

        public Params<T1, T2> SetParams<T3>(Params kwargs)
        {
            Set<Params>(nameof(kwargs), kwargs);
            return this;
        }
    }

    public class Params<T1, T2, T3> : Params
    {
        public static Params<T1, T2, T3> args<T1, T2, T3>(T1 arg1, T2 arg2, T3 arg3,
            [CallerArgumentExpression(nameof(arg1))] string? arg1Name = null,
            [CallerArgumentExpression(nameof(arg2))] string? arg2Name = null,
            [CallerArgumentExpression(nameof(arg3))] string? arg3Name = null)
        {
            var pc = new Params<T1, T2, T3>();
            pc.Set(arg1Name, arg1);
            pc.Set(arg2Name, arg2);
            pc.Set(arg3Name, arg3);
            return pc;
        }

        public Params<T1, T2, T3> SetParams(Params kwargs)
        {
            Set<Params>(nameof(kwargs), kwargs);
            return this;
        }
    }

    public class Params<T1, T2, T3, T4> : Params
    {
        public static Params<T1, T2, T3, T4> args<T1, T2, T3, T4>(T1 arg1, T2 arg2, T3 arg3, T4 arg4,
            [CallerArgumentExpression(nameof(arg1))] string? arg1Name = null,
            [CallerArgumentExpression(nameof(arg2))] string? arg2Name = null,
            [CallerArgumentExpression(nameof(arg3))] string? arg3Name = null,
            [CallerArgumentExpression(nameof(arg4))] string? arg4Name = null)
        {
            var pc = new Params<T1, T2, T3, T4>();
            pc.Set(arg1Name, arg1);
            pc.Set(arg2Name, arg2);
            pc.Set(arg3Name, arg3);
            pc.Set(arg4Name, arg4);
            return pc;
        }

        public Params<T1, T2, T3, T4> SetParams(Params kwargs)
        {
            Set<Params>(nameof(kwargs), kwargs);
            return this;
        }
    }

    public class Params<T1, T2, T3, T4, T5> : Params
    {
        public static Params<T1, T2, T3, T4, T5> args<T1, T2, T3, T4, T5>(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5,
            [CallerArgumentExpression(nameof(arg1))] string? arg1Name = null,
            [CallerArgumentExpression(nameof(arg2))] string? arg2Name = null,
            [CallerArgumentExpression(nameof(arg3))] string? arg3Name = null,
            [CallerArgumentExpression(nameof(arg4))] string? arg4Name = null,
            [CallerArgumentExpression(nameof(arg5))] string? arg5Name = null)
        {
            var pc = new Params<T1, T2, T3, T4, T5>();
            pc.Set(arg1Name, arg1);
            pc.Set(arg2Name, arg2);
            pc.Set(arg3Name, arg3);
            pc.Set(arg4Name, arg4);
            pc.Set(arg5Name, arg5);
            return pc;
        }

        public Params<T1, T2, T3, T4, T5> SetParams(Params kwargs)
        {
            Set<Params>(nameof(kwargs), kwargs);
            return this;
        }
    }

    public class Params<T1, T2, T3, T4, T5, T6> : Params
    {
        public static Params<T1, T2, T3, T4, T5, T6> args<T1, T2, T3, T4, T5, T6>(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6,
            [CallerArgumentExpression(nameof(arg1))] string? arg1Name = null,
            [CallerArgumentExpression(nameof(arg2))] string? arg2Name = null,
            [CallerArgumentExpression(nameof(arg3))] string? arg3Name = null,
            [CallerArgumentExpression(nameof(arg4))] string? arg4Name = null,
            [CallerArgumentExpression(nameof(arg5))] string? arg5Name = null,
            [CallerArgumentExpression(nameof(arg6))] string? arg6Name = null)
        {
            var pc = new Params<T1, T2, T3, T4, T5, T6>();
            pc.Set(arg1Name, arg1);
            pc.Set(arg2Name, arg2);
            pc.Set(arg3Name, arg3);
            pc.Set(arg4Name, arg4);
            pc.Set(arg5Name, arg5);
            pc.Set(arg6Name, arg6);
            return pc;
        }

        public Params<T1, T2, T3, T4, T5, T6> SetParams(Params kwargs)
        {
            Set<Params>(nameof(kwargs), kwargs);
            return this;
        }
    }

    public class Params<T1, T2, T3, T4, T5, T6, T7> : Params
    {
        public static Params<T1, T2, T3, T4, T5, T6, T7> args<T1, T2, T3, T4, T5, T6, T7>(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7,
            [CallerArgumentExpression(nameof(arg1))] string? arg1Name = null,
            [CallerArgumentExpression(nameof(arg2))] string? arg2Name = null,
            [CallerArgumentExpression(nameof(arg3))] string? arg3Name = null,
            [CallerArgumentExpression(nameof(arg4))] string? arg4Name = null,
            [CallerArgumentExpression(nameof(arg5))] string? arg5Name = null,
            [CallerArgumentExpression(nameof(arg6))] string? arg6Name = null,
            [CallerArgumentExpression(nameof(arg7))] string? arg7Name = null)
        {
            var pc = new Params<T1, T2, T3, T4, T5, T6, T7>();
            pc.Set(arg1Name, arg1);
            pc.Set(arg2Name, arg2);
            pc.Set(arg3Name, arg3);
            pc.Set(arg4Name, arg4);
            pc.Set(arg5Name, arg5);
            pc.Set(arg6Name, arg6);
            pc.Set(arg7Name, arg7);
            return pc;
        }

        public Params<T1, T2, T3, T4, T5, T6, T7> SetParams(Params kwargs)
        {
            Set<Params>(nameof(kwargs), kwargs);
            return this;
        }
    }

    public class Params<T1, T2, T3, T4, T5, T6, T7, T8> : Params
    {
        public static Params<T1, T2, T3, T4, T5, T6, T7, T8> args<T1, T2, T3, T4, T5, T6, T7, T8>(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8,
            [CallerArgumentExpression(nameof(arg1))] string? arg1Name = null,
            [CallerArgumentExpression(nameof(arg2))] string? arg2Name = null,
            [CallerArgumentExpression(nameof(arg3))] string? arg3Name = null,
            [CallerArgumentExpression(nameof(arg4))] string? arg4Name = null,
            [CallerArgumentExpression(nameof(arg5))] string? arg5Name = null,
            [CallerArgumentExpression(nameof(arg6))] string? arg6Name = null,
            [CallerArgumentExpression(nameof(arg7))] string? arg7Name = null,
            [CallerArgumentExpression(nameof(arg8))] string? arg8Name = null)
        {
            var pc = new Params<T1, T2, T3, T4, T5, T6, T7, T8>();
            pc.Set(arg1Name, arg1);
            pc.Set(arg2Name, arg2);
            pc.Set(arg3Name, arg3);
            pc.Set(arg4Name, arg4);
            pc.Set(arg5Name, arg5);
            pc.Set(arg6Name, arg6);
            pc.Set(arg7Name, arg7);
            pc.Set(arg8Name, arg8);
            return pc;
        }

        public Params<T1, T2, T3, T4, T5, T6, T7, T8> SetParams(Params kwargs)
        {
            Set<Params>(nameof(kwargs), kwargs);
            return this;
        }
    }
}
