
using System.Diagnostics;
using System.Runtime.CompilerServices;
using DeZero.NET;

namespace DeZero.NET.Core
{
    public class OrderedParams : Params
    {
        private int _index = 0;

        public override T Get<T>(string _)
        {
            try
            {
                var obj = _positional_args[_index++];
                return (T)(obj.Value is NDarray array ? array.ToVariable() : obj.Value);
            }
            catch (Exception e)
            {
                throw;
            }
        }

        public override T Get<T>(string _, T defaultValue)
        {
            try
            {
                var obj = _positional_args[_index++];
                return (T)(obj.Value is NDarray array ? array.ToVariable() : obj.Value);
            }
            catch (Exception e)
            {
                return defaultValue;
            }
        }

        public override Parameter[] Through => [.._positional_args];
    }

    public class OrderedParams<T1> : OrderedParams
    {
        public static OrderedParams<T1> args<T1>(T1 arg1, [CallerArgumentExpression(nameof(arg1))] string? arg1Name = null)
        {
            var pc = new OrderedParams<T1>();
            pc.SetKeywordArg(arg1Name, arg1);
            return pc;
        }

        [DebuggerStepThrough]
        public static OrderedParams<T1> args<T1>(T1[] args)
        {
            var pc = new OrderedParams<T1>();
            pc._positional_args.AddRange(args.Cast<object>().Select(x => new Parameter("@", x)));
            return pc;
        }
    }

    public class OrderedParams<T1, T2> : OrderedParams
    {
        public static OrderedParams<T1, T2> args<T1, T2>(T1 arg1, T2 arg2, [CallerArgumentExpression(nameof(arg1))] string? arg1Name = null, [CallerArgumentExpression(nameof(arg2))] string? arg2Name = null)
        {
            var pc = new OrderedParams<T1, T2>();
            pc.SetKeywordArg(arg1Name, arg1);
            pc.SetKeywordArg(arg2Name, arg2);
            return pc;
        }
    }

    public class OrderedParams<T1, T2, T3> : OrderedParams
    {
        public static OrderedParams<T1, T2, T3> args<T1, T2, T3>(T1 arg1, T2 arg2, T3 arg3,
            [CallerArgumentExpression(nameof(arg1))] string? arg1Name = null,
            [CallerArgumentExpression(nameof(arg2))] string? arg2Name = null,
            [CallerArgumentExpression(nameof(arg3))] string? arg3Name = null)
        {
            var pc = new OrderedParams<T1, T2, T3>();
            pc.SetKeywordArg(arg1Name, arg1);
            pc.SetKeywordArg(arg2Name, arg2);
            pc.SetKeywordArg(arg3Name, arg3);
            return pc;
        }
    }

    public class OrderedParams<T1, T2, T3, T4> : OrderedParams
    {
        public static OrderedParams<T1, T2, T3, T4> args<T1, T2, T3, T4>(T1 arg1, T2 arg2, T3 arg3, T4 arg4,
            [CallerArgumentExpression(nameof(arg1))] string? arg1Name = null,
            [CallerArgumentExpression(nameof(arg2))] string? arg2Name = null,
            [CallerArgumentExpression(nameof(arg3))] string? arg3Name = null,
            [CallerArgumentExpression(nameof(arg4))] string? arg4Name = null)
        {
            var pc = new OrderedParams<T1, T2, T3, T4>();
            pc.SetKeywordArg(arg1Name, arg1);
            pc.SetKeywordArg(arg2Name, arg2);
            pc.SetKeywordArg(arg3Name, arg3);
            pc.SetKeywordArg(arg4Name, arg4);
            return pc;
        }
    }

    public class OrderedParams<T1, T2, T3, T4, T5> : OrderedParams
    {
        public static OrderedParams<T1, T2, T3, T4, T5> args<T1, T2, T3, T4, T5>(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5,
            [CallerArgumentExpression(nameof(arg1))] string? arg1Name = null,
            [CallerArgumentExpression(nameof(arg2))] string? arg2Name = null,
            [CallerArgumentExpression(nameof(arg3))] string? arg3Name = null,
            [CallerArgumentExpression(nameof(arg4))] string? arg4Name = null,
            [CallerArgumentExpression(nameof(arg5))] string? arg5Name = null)
        {
            var pc = new OrderedParams<T1, T2, T3, T4, T5>();
            pc.SetKeywordArg(arg1Name, arg1);
            pc.SetKeywordArg(arg2Name, arg2);
            pc.SetKeywordArg(arg3Name, arg3);
            pc.SetKeywordArg(arg4Name, arg4);
            pc.SetKeywordArg(arg5Name, arg5);
            return pc;
        }
    }

    public class OrderedParams<T1, T2, T3, T4, T5, T6> : OrderedParams
    {
        public static OrderedParams<T1, T2, T3, T4, T5, T6> args<T1, T2, T3, T4, T5, T6>(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6,
            [CallerArgumentExpression(nameof(arg1))] string? arg1Name = null,
            [CallerArgumentExpression(nameof(arg2))] string? arg2Name = null,
            [CallerArgumentExpression(nameof(arg3))] string? arg3Name = null,
            [CallerArgumentExpression(nameof(arg4))] string? arg4Name = null,
            [CallerArgumentExpression(nameof(arg5))] string? arg5Name = null,
            [CallerArgumentExpression(nameof(arg6))] string? arg6Name = null)
        {
            var pc = new OrderedParams<T1, T2, T3, T4, T5, T6>();
            pc.SetKeywordArg(arg1Name, arg1);
            pc.SetKeywordArg(arg2Name, arg2);
            pc.SetKeywordArg(arg3Name, arg3);
            pc.SetKeywordArg(arg4Name, arg4);
            pc.SetKeywordArg(arg5Name, arg5);
            pc.SetKeywordArg(arg6Name, arg6);
            return pc;
        }
    }
}
