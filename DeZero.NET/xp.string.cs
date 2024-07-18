using Cupy;
using Numpy;

namespace DeZero.NET
{
    public static partial class xp
    {
        public static partial class core
        {
            public static partial class defchararray
            {
                /// <summary>
                ///     Return element-wise string concatenation for two arrays of str or unicode.<br></br>
                ///     Arrays x1 and x2 must have the same shape.
                /// </summary>
                /// <param name="x1">
                ///     Input array.
                /// </param>
                /// <param name="x2">
                ///     Input array.
                /// </param>
                /// <returns>
                ///     Output array of string_ or unicode_, depending on input types
                ///     of the same shape as x1 and x2.
                /// </returns>
                public static NDarray add(string[] x1, string[] x2)
                {
                    if (Gpu.Available && Gpu.Use)
                    {
                        return new NDarray(cp.add(x1, x2));
                    }
                    else
                    {
                        return new NDarray(np.add(x1, x2));
                    }
                }
            }
        }

        public static partial class core
        {
            public static partial class defchararray
            {
                /// <summary>
                ///     Return (a * i), that is string multiple concatenation,
                ///     element-wise.<br></br>
                ///     Values in i of less than 0 are treated as 0 (which yields an
                ///     empty string).
                /// </summary>
                /// <returns>
                ///     Output array of str or unicode, depending on input types
                /// </returns>
                public static NDarray multiply(string[] a, int[] i)
                {
                    if (Gpu.Available && Gpu.Use)
                    {
                        return new NDarray(cp.multiply(a, i));
                    }
                    else
                    {
                        return new NDarray(np.multiply(a, i));
                    }
                }
            }
        }

        public static partial class core
        {
            public static partial class defchararray
            {
                /// <summary>
                ///     Return (a % i), that is pre-Python 2.6 string formatting
                ///     (iterpolation), element-wise for a pair of array_likes of str
                ///     or unicode.
                /// </summary>
                /// <param name="values">
                ///     These values will be element-wise interpolated into the string.
                /// </param>
                /// <returns>
                ///     Output array of str or unicode, depending on input types
                /// </returns>
                public static NDarray mod(string[] a, NDarray values)
                {
                    if (Gpu.Available && Gpu.Use)
                    {
                        return new NDarray(cp.mod(a, values.CupyNDarray));
                    }
                    else
                    {
                        return new NDarray(np.mod(a, values.NumpyNDarray));
                    }
                }
            }
        }

        public static partial class core
        {
            public static partial class defchararray
            {
                /// <summary>
                ///     Return a copy of a with only the first character of each element
                ///     capitalized.<br></br>
                ///     Calls str.capitalize element-wise.<br></br>
                ///     For 8-bit strings, this method is locale-dependent.
                /// </summary>
                /// <param name="a">
                ///     Input array of strings to capitalize.
                /// </param>
                /// <returns>
                ///     Output array of str or unicode, depending on input
                ///     types
                /// </returns>
                public static NDarray capitalize(params string[] a)
                {
                    if (Gpu.Available && Gpu.Use)
                    {
                        return new NDarray(cp.core.defchararray.capitalize(a));
                    }
                    else
                    {
                        return new NDarray(np.core.defchararray.capitalize(a));
                    }
                }
            }
        }

        public static partial class core
        {
            public static partial class defchararray
            {
                /// <summary>
                ///     Return a copy of a with its elements centered in a string of
                ///     length width.<br></br>
                ///     Calls str.center element-wise.
                /// </summary>
                /// <param name="width">
                ///     The length of the resulting strings
                /// </param>
                /// <param name="fillchar">
                ///     The padding character to use (default is space).
                /// </param>
                /// <returns>
                ///     Output array of str or unicode, depending on input
                ///     types
                /// </returns>
                public static NDarray center(string[] a, int width, string fillchar = " ")
                {
                    if (Gpu.Available && Gpu.Use)
                    {
                        return new NDarray(cp.core.defchararray.center(a, width, fillchar));
                    }
                    else
                    {
                        return new NDarray(np.core.defchararray.center(a, width, fillchar));
                    }
                }
            }
        }

        public static partial class core
        {
            public static partial class defchararray
            {
                /// <summary>
                ///     Calls str.decode element-wise.<br></br>
                ///     The set of available codecs comes from the Python standard library,
                ///     and may be extended at runtime.<br></br>
                ///     For more information, see the
                ///     codecs module.<br></br>
                ///     Notes
                ///     The type of the result will depend on the encoding specified.
                /// </summary>
                /// <param name="encoding">
                ///     The name of an encoding
                /// </param>
                /// <param name="errors">
                ///     Specifies how to handle encoding errors
                /// </param>
                public static NDarray decode(string[] a, string encoding = null, string errors = null)
                {
                    if (Gpu.Available && Gpu.Use)
                    {
                        return new NDarray(cp.core.defchararray.decode(a, encoding, errors));
                    }
                    else
                    {
                        return new NDarray(np.core.defchararray.decode(a, encoding, errors));
                    }
                }
            }
        }

        public static partial class core
        {
            public static partial class defchararray
            {
                /// <summary>
                ///     Calls str.encode element-wise.<br></br>
                ///     The set of available codecs comes from the Python standard library,
                ///     and may be extended at runtime.<br></br>
                ///     For more information, see the codecs
                ///     module.<br></br>
                ///     Notes
                ///     The type of the result will depend on the encoding specified.
                /// </summary>
                /// <param name="encoding">
                ///     The name of an encoding
                /// </param>
                /// <param name="errors">
                ///     Specifies how to handle encoding errors
                /// </param>
                public static NDarray encode(string[] a, string encoding = null, string errors = null)
                {
                    if (Gpu.Available && Gpu.Use)
                    {
                        return new NDarray(cp.core.defchararray.encode(a, encoding, errors));
                    }
                    else
                    {
                        return new NDarray(np.core.defchararray.encode(a, encoding, errors));
                    }
                }
            }
        }

        public static partial class core
        {
            public static partial class defchararray
            {
                /// <summary>
                ///     Return a string which is the concatenation of the strings in the
                ///     sequence seq.<br></br>
                ///     Calls str.join element-wise.
                /// </summary>
                /// <returns>
                ///     Output array of str or unicode, depending on input types
                /// </returns>
                public static NDarray join(string[] sep, string[] seq)
                {
                    if (Gpu.Available && Gpu.Use)
                    {
                        return new NDarray(cp.core.defchararray.join(sep, seq));
                    }
                    else
                    {
                        return new NDarray(np.core.defchararray.join(sep, seq));
                    }
                }
            }
        }

        public static partial class core
        {
            public static partial class defchararray
            {
                /// <summary>
                ///     Return an array with the elements of a left-justified in a
                ///     string of length width.<br></br>
                ///     Calls str.ljust element-wise.
                /// </summary>
                /// <param name="width">
                ///     The length of the resulting strings
                /// </param>
                /// <param name="fillchar">
                ///     The character to use for padding
                /// </param>
                /// <returns>
                ///     Output array of str or unicode, depending on input type
                /// </returns>
                public static NDarray ljust(string[] a, int width, string fillchar = " ")
                {
                    if (Gpu.Available && Gpu.Use)
                    {
                        return new NDarray(cp.core.defchararray.ljust(a, width, fillchar));
                    }
                    else
                    {
                        return new NDarray(np.core.defchararray.ljust(a, width, fillchar));
                    }
                }
            }
        }

        public static partial class core
        {
            public static partial class defchararray
            {
                /// <summary>
                ///     Return an array with the elements converted to lowercase.<br></br>
                ///     Call str.lower element-wise.<br></br>
                ///     For 8-bit strings, this method is locale-dependent.
                /// </summary>
                /// <param name="a">
                ///     Input array.
                /// </param>
                /// <returns>
                ///     Output array of str or unicode, depending on input type
                /// </returns>
                public static NDarray lower(NDarray a)
                {
                    if (Gpu.Available && Gpu.Use)
                    {
                        return new NDarray(cp.core.defchararray.lower(a.CupyNDarray));
                    }
                    else
                    {
                        return new NDarray(np.core.defchararray.lower(a.NumpyNDarray));
                    }
                }
            }
        }

        public static partial class core
        {
            public static partial class defchararray
            {
                /// <summary>
                ///     For each element in a, return a copy with the leading characters
                ///     removed.<br></br>
                ///     Calls str.lstrip element-wise.
                /// </summary>
                /// <param name="a">
                ///     Input array.
                /// </param>
                /// <param name="chars">
                ///     The chars argument is a string specifying the set of
                ///     characters to be removed.<br></br>
                ///     If omitted or None, the chars
                ///     argument defaults to removing whitespace.<br></br>
                ///     The chars argument
                ///     is not a prefix; rather, all combinations of its values are
                ///     stripped.
                /// </param>
                /// <returns>
                ///     Output array of str or unicode, depending on input type
                /// </returns>
                public static NDarray lstrip(NDarray a, string chars = null)
                {
                    if (Gpu.Available && Gpu.Use)
                    {
                        return new NDarray(cp.core.defchararray.lstrip(a.CupyNDarray, chars));
                    }
                    else
                    {
                        return new NDarray(np.core.defchararray.lstrip(a.NumpyNDarray, chars));
                    }
                }
            }
        }

        public static partial class core
        {
            public static partial class defchararray
            {
                /// <summary>
                ///     Partition each element in a around sep.<br></br>
                ///     Calls str.partition element-wise.<br></br>
                ///     For each element in a, split the element as the first
                ///     occurrence of sep, and return 3 strings containing the part
                ///     before the separator, the separator itself, and the part after
                ///     the separator.<br></br>
                ///     If the separator is not found, return 3 strings
                ///     containing the string itself, followed by two empty strings.
                /// </summary>
                /// <param name="a">
                ///     Input array
                /// </param>
                /// <param name="sep">
                ///     Separator to split each string element in a.
                /// </param>
                /// <returns>
                ///     Output array of str or unicode, depending on input type.<br></br>
                ///     The output array will have an extra dimension with 3
                ///     elements per input element.
                /// </returns>
                public static NDarray partition(NDarray a, string sep)
                {
                    if (Gpu.Available && Gpu.Use)
                    {
                        return new NDarray(cp.core.defchararray.partition(a.CupyNDarray, sep));
                    }
                    else
                    {
                        return new NDarray(np.core.defchararray.partition(a.NumpyNDarray, sep));
                    }
                }
            }
        }

        public static partial class core
        {
            public static partial class defchararray
            {
                /// <summary>
                ///     For each element in a, return a copy of the string with all
                ///     occurrences of substring old replaced by new.<br></br>
                ///     Calls str.replace element-wise.
                /// </summary>
                /// <param name="count">
                ///     If the optional argument count is given, only the first
                ///     count occurrences are replaced.
                /// </param>
                /// <returns>
                ///     Output array of str or unicode, depending on input type
                /// </returns>
                public static NDarray replace(string[] a, string @new, string old, int? count = null)
                {
                    if (Gpu.Available && Gpu.Use)
                    {
                        return new NDarray(cp.core.defchararray.replace(a, @new, old, count));
                    }
                    else
                    {
                        return new NDarray(np.core.defchararray.replace(a, @new, old, count));
                    }
                }
            }
        }

        public static partial class core
        {
            public static partial class defchararray
            {
                /// <summary>
                ///     Return an array with the elements of a right-justified in a
                ///     string of length width.<br></br>
                ///     Calls str.rjust element-wise.
                /// </summary>
                /// <param name="width">
                ///     The length of the resulting strings
                /// </param>
                /// <param name="fillchar">
                ///     The character to use for padding
                /// </param>
                /// <returns>
                ///     Output array of str or unicode, depending on input type
                /// </returns>
                public static NDarray rjust(string[] a, int width, string fillchar = " ")
                {
                    if (Gpu.Available && Gpu.Use)
                    {
                        return new NDarray(cp.core.defchararray.rjust(a, width, fillchar));
                    }
                    else
                    {
                        return new NDarray(np.core.defchararray.rjust(a, width, fillchar));
                    }
                }
            }
        }

        public static partial class core
        {
            public static partial class defchararray
            {
                /// <summary>
                ///     Partition (split) each element around the right-most separator.<br></br>
                ///     Calls str.rpartition element-wise.<br></br>
                ///     For each element in a, split the element as the last
                ///     occurrence of sep, and return 3 strings containing the part
                ///     before the separator, the separator itself, and the part after
                ///     the separator.<br></br>
                ///     If the separator is not found, return 3 strings
                ///     containing the string itself, followed by two empty strings.
                /// </summary>
                /// <param name="a">
                ///     Input array
                /// </param>
                /// <param name="sep">
                ///     Right-most separator to split each element in array.
                /// </param>
                /// <returns>
                ///     Output array of string or unicode, depending on input
                ///     type.<br></br>
                ///     The output array will have an extra dimension with
                ///     3 elements per input element.
                /// </returns>
                public static NDarray rpartition(string[] a, string sep)
                {
                    if (Gpu.Available && Gpu.Use)
                    {
                        return new NDarray(cp.core.defchararray.rpartition(a, sep));
                    }
                    else
                    {
                        return new NDarray(np.core.defchararray.rpartition(a, sep));
                    }
                }
            }
        }

        public static partial class core
        {
            public static partial class defchararray
            {
                /// <summary>
                ///     For each element in a, return a list of the words in the
                ///     string, using sep as the delimiter string.<br></br>
                ///     Calls str.rsplit element-wise.<br></br>
                ///     Except for splitting from the right, rsplit
                ///     behaves like split.
                /// </summary>
                /// <param name="sep">
                ///     If sep is not specified or None, any whitespace string
                ///     is a separator.
                /// </param>
                /// <param name="maxsplit">
                ///     If maxsplit is given, at most maxsplit splits are done,
                ///     the rightmost ones.
                /// </param>
                /// <returns>
                ///     Array of list objects
                /// </returns>
                public static NDarray rsplit(string[] a, string sep = null, int? maxsplit = null)
                {
                    if (Gpu.Available && Gpu.Use)
                    {
                        return new NDarray(cp.core.defchararray.rsplit(a, sep, maxsplit));
                    }
                    else
                    {
                        return new NDarray(np.core.defchararray.rsplit(a, sep, maxsplit));
                    }
                }
            }
        }

        public static partial class core
        {
            public static partial class defchararray
            {
                /// <summary>
                ///     For each element in a, return a copy with the trailing
                ///     characters removed.<br></br>
                ///     Calls str.rstrip element-wise.
                /// </summary>
                /// <param name="chars">
                ///     The chars argument is a string specifying the set of
                ///     characters to be removed.<br></br>
                ///     If omitted or None, the chars
                ///     argument defaults to removing whitespace.<br></br>
                ///     The chars argument
                ///     is not a suffix; rather, all combinations of its values are
                ///     stripped.
                /// </param>
                /// <returns>
                ///     Output array of str or unicode, depending on input type
                /// </returns>
                public static NDarray rstrip(string[] a, string chars = null)
                {
                    if (Gpu.Available && Gpu.Use)
                    {
                        return new NDarray(cp.core.defchararray.rstrip(a, chars));
                    }
                    else
                    {
                        return new NDarray(np.core.defchararray.rstrip(a, chars));
                    }
                }
            }
        }

        public static partial class core
        {
            public static partial class defchararray
            {
                /// <summary>
                ///     For each element in a, return a list of the words in the
                ///     string, using sep as the delimiter string.<br></br>
                ///     Calls str.split element-wise.
                /// </summary>
                /// <param name="sep">
                ///     If sep is not specified or None, any whitespace string is a
                ///     separator.
                /// </param>
                /// <param name="maxsplit">
                ///     If maxsplit is given, at most maxsplit splits are done.
                /// </param>
                /// <returns>
                ///     Array of list objects
                /// </returns>
                public static NDarray split(string[] a, string sep = null, int? maxsplit = null)
                {
                    if (Gpu.Available && Gpu.Use)
                    {
                        return new NDarray(cp.core.defchararray.split(a, sep, maxsplit));
                    }
                    else
                    {
                        return new NDarray(np.core.defchararray.split(a, sep, maxsplit));
                    }
                }
            }
        }

        public static partial class core
        {
            public static partial class defchararray
            {
                /// <summary>
                ///     For each element in a, return a list of the words in the
                ///     string, using sep as the delimiter string.<br></br>
                ///     Calls str.split element-wise.
                /// </summary>
                /// <param name="sep">
                ///     If sep is not specified or None, any whitespace string is a
                ///     separator.
                /// </param>
                /// <param name="maxsplit">
                ///     If maxsplit is given, at most maxsplit splits are done.
                /// </param>
                /// <returns>
                ///     Array of list objects
                /// </returns>
                public static NDarray split(string[] a, int? sep = null, int? maxsplit = null)
                {
                    if (Gpu.Available && Gpu.Use)
                    {
                        return new NDarray(cp.core.defchararray.split(a, sep, maxsplit));
                    }
                    else
                    {
                        return new NDarray(np.core.defchararray.split(a, sep?.ToString(), maxsplit));
                    }
                }
            }
        }

        public static partial class core
        {
            public static partial class defchararray
            {
                /// <summary>
                ///     For each element in a, return a list of the lines in the
                ///     element, breaking at line boundaries.<br></br>
                ///     Calls str.splitlines element-wise.
                /// </summary>
                /// <param name="keepends">
                ///     Line breaks are not included in the resulting list unless
                ///     keepends is given and true.
                /// </param>
                /// <returns>
                ///     Array of list objects
                /// </returns>
                public static NDarray splitlines(string[] a, bool? keepends = null)
                {
                    if (Gpu.Available && Gpu.Use)
                    {
                        return new NDarray(cp.core.defchararray.splitlines(a, keepends));
                    }
                    else
                    {
                        return new NDarray(np.core.defchararray.splitlines(a, keepends));
                    }
                }
            }
        }

        public static partial class core
        {
            public static partial class defchararray
            {
                /// <summary>
                ///     For each element in a, return a copy with the leading and
                ///     trailing characters removed.<br></br>
                ///     Calls str.strip element-wise.
                /// </summary>
                /// <param name="chars">
                ///     The chars argument is a string specifying the set of
                ///     characters to be removed.<br></br>
                ///     If omitted or None, the chars
                ///     argument defaults to removing whitespace.<br></br>
                ///     The chars argument
                ///     is not a prefix or suffix; rather, all combinations of its
                ///     values are stripped.
                /// </param>
                /// <returns>
                ///     Output array of str or unicode, depending on input type
                /// </returns>
                public static NDarray strip(string[] a, string chars = null)
                {
                    if (Gpu.Available && Gpu.Use)
                    {
                        return new NDarray(cp.core.defchararray.strip(a, chars));
                    }
                    else
                    {
                        return new NDarray(np.core.defchararray.strip(a, chars));
                    }
                }
            }
        }

        public static partial class core
        {
            public static partial class defchararray
            {
                /// <summary>
                ///     Return element-wise a copy of the string with
                ///     uppercase characters converted to lowercase and vice versa.<br></br>
                ///     Calls str.swapcase element-wise.<br></br>
                ///     For 8-bit strings, this method is locale-dependent.
                /// </summary>
                /// <param name="a">
                ///     Input array.
                /// </param>
                /// <returns>
                ///     Output array of str or unicode, depending on input type
                /// </returns>
                public static NDarray swapcase(NDarray a)
                {
                    if (Gpu.Available && Gpu.Use)
                    {
                        return new NDarray(cp.core.defchararray.swapcase(a.CupyNDarray));
                    }
                    else
                    {
                        return new NDarray(np.core.defchararray.swapcase(a.NumpyNDarray));
                    }
                }
            }
        }

        public static partial class core
        {
            public static partial class defchararray
            {
                /// <summary>
                ///     Return element-wise title cased version of string or unicode.<br></br>
                ///     Title case words start with uppercase characters, all remaining cased
                ///     characters are lowercase.<br></br>
                ///     Calls str.title element-wise.<br></br>
                ///     For 8-bit strings, this method is locale-dependent.
                /// </summary>
                /// <param name="a">
                ///     Input array.
                /// </param>
                /// <returns>
                ///     Output array of str or unicode, depending on input type
                /// </returns>
                public static NDarray title(NDarray a)
                {
                    if (Gpu.Available && Gpu.Use)
                    {
                        return new NDarray(cp.core.defchararray.title(a.CupyNDarray));
                    }
                    else
                    {
                        return new NDarray(np.core.defchararray.title(a.NumpyNDarray));
                    }
                }
            }
        }

        public static partial class core
        {
            public static partial class defchararray
            {
                /// <summary>
                ///     For each element in a, return a copy of the string where all
                ///     characters occurring in the optional argument deletechars are
                ///     removed, and the remaining characters have been mapped through the
                ///     given translation table.<br></br>
                ///     Calls str.translate element-wise.
                /// </summary>
                /// <returns>
                ///     Output array of str or unicode, depending on input type
                /// </returns>
                public static NDarray translate(string[] a, string table, string deletechars)
                {
                    if (Gpu.Available && Gpu.Use)
                    {
                        return new NDarray(cp.core.defchararray.translate(a, table, deletechars));
                    }
                    else
                    {
                        return new NDarray(np.core.defchararray.translate(a, table, deletechars));
                    }
                }
            }
        }

        public static partial class core
        {
            public static partial class defchararray
            {
                /// <summary>
                ///     Return an array with the elements converted to uppercase.<br></br>
                ///     Calls str.upper element-wise.<br></br>
                ///     For 8-bit strings, this method is locale-dependent.
                /// </summary>
                /// <param name="a">
                ///     Input array.
                /// </param>
                /// <returns>
                ///     Output array of str or unicode, depending on input type
                /// </returns>
                public static NDarray upper(NDarray a)
                {
                    if (Gpu.Available && Gpu.Use)
                    {
                        return new NDarray(cp.core.defchararray.upper(a.CupyNDarray));
                    }
                    else
                    {
                        return new NDarray(np.core.defchararray.upper(a.NumpyNDarray));
                    }
                }
            }
        }

        public static partial class core
        {
            public static partial class defchararray
            {
                /// <summary>
                ///     Return the numeric string left-filled with zeros
                ///     Calls str.zfill element-wise.
                /// </summary>
                /// <param name="a">
                ///     Input array.
                /// </param>
                /// <param name="width">
                ///     Width of string to left-fill elements in a.
                /// </param>
                /// <returns>
                ///     Output array of str or unicode, depending on input type
                /// </returns>
                public static NDarray zfill(NDarray a, int width)
                {
                    if (Gpu.Available && Gpu.Use)
                    {
                        return new NDarray(cp.core.defchararray.zfill(a.CupyNDarray, width));
                    }
                    else
                    {
                        return new NDarray(np.core.defchararray.zfill(a.NumpyNDarray, width));
                    }
                }
            }
        }

        public static partial class core
        {
            public static partial class defchararray
            {
                /// <summary>
                ///     Return (x1 == x2) element-wise.<br></br>
                ///     Unlike Cupy.equal, this comparison is performed by first
                ///     stripping whitespace characters from the end of the string.<br></br>
                ///     This
                ///     behavior is provided for backward-compatibility with numarray.
                /// </summary>
                /// <param name="x2">
                ///     Input arrays of the same shape.
                /// </param>
                /// <param name="x1">
                ///     Input arrays of the same shape.
                /// </param>
                /// <returns>
                ///     Output array of bools, or a single bool if x1 and x2 are scalars.
                /// </returns>
                public static NDarray equal(string[] x2, string[] x1)
                {
                    if (Gpu.Available && Gpu.Use)
                    {
                        return new NDarray(cp.core.defchararray.equal(x2, x1));
                    }
                    else
                    {
                        return new NDarray(np.core.defchararray.equal(x2, x1));
                    }
                }
            }
        }

        public static partial class core
        {
            public static partial class defchararray
            {
                /// <summary>
                ///     Return (x1 != x2) element-wise.<br></br>
                ///     Unlike Cupy.not_equal, this comparison is performed by first
                ///     stripping whitespace characters from the end of the string.<br></br>
                ///     This
                ///     behavior is provided for backward-compatibility with numarray.
                /// </summary>
                /// <param name="x2">
                ///     Input arrays of the same shape.
                /// </param>
                /// <param name="x1">
                ///     Input arrays of the same shape.
                /// </param>
                /// <returns>
                ///     Output array of bools, or a single bool if x1 and x2 are scalars.
                /// </returns>
                public static NDarray not_equal(string[] x2, string[] x1)
                {
                    if (Gpu.Available && Gpu.Use)
                    {
                        return new NDarray(cp.core.defchararray.not_equal(x2, x1));
                    }
                    else
                    {
                        return new NDarray(np.core.defchararray.not_equal(x2, x1));
                    }
                }
            }
        }

        public static partial class core
        {
            public static partial class defchararray
            {
                /// <summary>
                ///     Return (x1 &gt;= x2) element-wise.<br></br>
                ///     Unlike Cupy.greater_equal, this comparison is performed by
                ///     first stripping whitespace characters from the end of the string.<br></br>
                ///     This behavior is provided for backward-compatibility with
                ///     numarray.
                /// </summary>
                /// <param name="x2">
                ///     Input arrays of the same shape.
                /// </param>
                /// <param name="x1">
                ///     Input arrays of the same shape.
                /// </param>
                /// <returns>
                ///     Output array of bools, or a single bool if x1 and x2 are scalars.
                /// </returns>
                public static NDarray greater_equal(string[] x2, string[] x1)
                {
                    if (Gpu.Available && Gpu.Use)
                    {
                        return new NDarray(cp.core.defchararray.greater_equal(x2, x1));
                    }
                    else
                    {
                        return new NDarray(np.core.defchararray.greater_equal(x2, x1));
                    }
                }
            }
        }

        public static partial class core
        {
            public static partial class defchararray
            {
                /// <summary>
                ///     Return (x1 &lt;= x2) element-wise.<br></br>
                ///     Unlike Cupy.less_equal, this comparison is performed by first
                ///     stripping whitespace characters from the end of the string.<br></br>
                ///     This
                ///     behavior is provided for backward-compatibility with numarray.
                /// </summary>
                /// <param name="x2">
                ///     Input arrays of the same shape.
                /// </param>
                /// <param name="x1">
                ///     Input arrays of the same shape.
                /// </param>
                /// <returns>
                ///     Output array of bools, or a single bool if x1 and x2 are scalars.
                /// </returns>
                public static NDarray less_equal(string[] x2, string[] x1)
                {
                    if (Gpu.Available && Gpu.Use)
                    {
                        return new NDarray(cp.core.defchararray.less_equal(x2, x1));
                    }
                    else
                    {
                        return new NDarray(np.core.defchararray.less_equal(x2, x1));
                    }
                }
            }
        }

        public static partial class core
        {
            public static partial class defchararray
            {
                /// <summary>
                ///     Return (x1 &gt; x2) element-wise.<br></br>
                ///     Unlike Cupy.greater, this comparison is performed by first
                ///     stripping whitespace characters from the end of the string.<br></br>
                ///     This
                ///     behavior is provided for backward-compatibility with numarray.
                /// </summary>
                /// <param name="x2">
                ///     Input arrays of the same shape.
                /// </param>
                /// <param name="x1">
                ///     Input arrays of the same shape.
                /// </param>
                /// <returns>
                ///     Output array of bools, or a single bool if x1 and x2 are scalars.
                /// </returns>
                public static NDarray greater(string[] x2, string[] x1)
                {
                    if (Gpu.Available && Gpu.Use)
                    {
                        return new NDarray(cp.core.defchararray.greater(x2, x1));
                    }
                    else
                    {
                        return new NDarray(np.core.defchararray.greater(x2, x1));
                    }
                }
            }
        }

        public static partial class core
        {
            public static partial class defchararray
            {
                /// <summary>
                ///     Return (x1 &lt; x2) element-wise.<br></br>
                ///     Unlike Cupy.greater, this comparison is performed by first
                ///     stripping whitespace characters from the end of the string.<br></br>
                ///     This
                ///     behavior is provided for backward-compatibility with numarray.
                /// </summary>
                /// <param name="x2">
                ///     Input arrays of the same shape.
                /// </param>
                /// <param name="x1">
                ///     Input arrays of the same shape.
                /// </param>
                /// <returns>
                ///     Output array of bools, or a single bool if x1 and x2 are scalars.
                /// </returns>
                public static NDarray less(string[] x2, string[] x1)
                {
                    if (Gpu.Available && Gpu.Use)
                    {
                        return new NDarray(cp.core.defchararray.less(x2, x1));
                    }
                    else
                    {
                        return new NDarray(np.core.defchararray.less(x2, x1));
                    }
                }
            }
        }

        public static partial class core
        {
            public static partial class defchararray
            {
                /// <summary>
                ///     Returns an array with the number of non-overlapping occurrences of
                ///     substring sub in the range [start, end].<br></br>
                ///     Calls str.count element-wise.
                /// </summary>
                /// <param name="sub">
                ///     The substring to search for.
                /// </param>
                /// <param name="end">
                ///     Optional arguments start and end are interpreted as slice
                ///     notation to specify the range in which to count.
                /// </param>
                /// <param name="start">
                ///     Optional arguments start and end are interpreted as slice
                ///     notation to specify the range in which to count.
                /// </param>
                /// <returns>
                ///     Output array of ints.
                /// </returns>
                public static NDarray count(string[] a, string sub, int? end = null, int? start = null)
                {
                    if (Gpu.Available && Gpu.Use)
                    {
                        return new NDarray(cp.core.defchararray.count(a, sub, end, start));
                    }
                    else
                    {
                        return new NDarray(np.core.defchararray.count(a, sub, end, start));
                    }
                }
            }
        }

        public static partial class core
        {
            public static partial class defchararray
            {
                /// <summary>
                ///     For each element, return the lowest index in the string where
                ///     substring sub is found.<br></br>
                ///     Calls str.find element-wise.<br></br>
                ///     For each element, return the lowest index in the string where
                ///     substring sub is found, such that sub is contained in the
                ///     range [start, end].
                /// </summary>
                /// <param name="end">
                ///     Optional arguments start and end are interpreted as in
                ///     slice notation.
                /// </param>
                /// <param name="start">
                ///     Optional arguments start and end are interpreted as in
                ///     slice notation.
                /// </param>
                /// <returns>
                ///     Output array of ints.<br></br>
                ///     Returns -1 if sub is not found.
                /// </returns>
                public static NDarray find(string[] a, string sub, int? end = null, int? start = null)
                {
                    if (Gpu.Available && Gpu.Use)
                    {
                        return new NDarray(cp.core.defchararray.find(a, sub, end, start));
                    }
                    else
                    {
                        return new NDarray(np.core.defchararray.find(a, sub, end, start));
                    }
                }
            }
        }

        public static partial class core
        {
            public static partial class defchararray
            {
                /// <summary>
                ///     Like find, but raises ValueError when the substring is not found.<br></br>
                ///     Calls str.index element-wise.
                /// </summary>
                /// <returns>
                ///     Output array of ints.<br></br>
                ///     Returns -1 if sub is not found.
                /// </returns>
                public static NDarray index(string[] a, string sub, int? end = null, int? start = null)
                {
                    if (Gpu.Available && Gpu.Use)
                    {
                        return new NDarray(cp.core.defchararray.index(a, sub, end, start));
                    }
                    else
                    {
                        return new NDarray(np.core.defchararray.index(a, sub, end, start));
                    }
                }
            }
        }

        public static partial class core
        {
            public static partial class defchararray
            {
                /// <summary>
                ///     Returns true for each element if all characters in the string are
                ///     alphabetic and there is at least one character, false otherwise.<br></br>
                ///     Calls str.isalpha element-wise.<br></br>
                ///     For 8-bit strings, this method is locale-dependent.
                /// </summary>
                /// <returns>
                ///     Output array of bools
                /// </returns>
                public static NDarray isalpha(params string[] a)
                {
                    if (Gpu.Available && Gpu.Use)
                    {
                        return new NDarray(cp.core.defchararray.isalpha(a));
                    }
                    else
                    {
                        return new NDarray(np.core.defchararray.isalpha(a));
                    }
                }
            }
        }

        public static partial class core
        {
            public static partial class defchararray
            {
                /// <summary>
                ///     For each element, return True if there are only decimal
                ///     characters in the element.<br></br>
                ///     Calls unicode.isdecimal element-wise.<br></br>
                ///     Decimal characters include digit characters, and all characters
                ///     that that can be used to form decimal-radix numbers,
                ///     e.g.<br></br>
                ///     U+0660, ARABIC-INDIC DIGIT ZERO.
                /// </summary>
                /// <param name="a">
                ///     Input array.
                /// </param>
                /// <returns>
                ///     Array of booleans identical in shape to a.
                /// </returns>
                public static NDarray isdecimal(NDarray a)
                {
                    if (Gpu.Available && Gpu.Use)
                    {
                        return new NDarray(cp.core.defchararray.isdecimal(a.CupyNDarray));
                    }
                    else
                    {
                        return new NDarray(np.core.defchararray.isdecimal(a.NumpyNDarray));
                    }
                }
            }
        }

        public static partial class core
        {
            public static partial class defchararray
            {
                /// <summary>
                ///     Returns true for each element if all characters in the string are
                ///     digits and there is at least one character, false otherwise.<br></br>
                ///     Calls str.isdigit element-wise.<br></br>
                ///     For 8-bit strings, this method is locale-dependent.
                /// </summary>
                /// <returns>
                ///     Output array of bools
                /// </returns>
                public static NDarray isdigit(params string[] a)
                {
                    if (Gpu.Available && Gpu.Use)
                    {
                        return new NDarray(cp.core.defchararray.isdigit(a));
                    }
                    else
                    {
                        return new NDarray(np.core.defchararray.isdigit(a));
                    }
                }
            }
        }

        public static partial class core
        {
            public static partial class defchararray
            {
                /// <summary>
                ///     Returns true for each element if all cased characters in the
                ///     string are lowercase and there is at least one cased character,
                ///     false otherwise.<br></br>
                ///     Calls str.islower element-wise.<br></br>
                ///     For 8-bit strings, this method is locale-dependent.
                /// </summary>
                /// <returns>
                ///     Output array of bools
                /// </returns>
                public static NDarray islower(params string[] a)
                {
                    if (Gpu.Available && Gpu.Use)
                    {
                        return new NDarray(cp.core.defchararray.islower(a));
                    }
                    else
                    {
                        return new NDarray(np.core.defchararray.islower(a));
                    }
                }
            }
        }

        public static partial class core
        {
            public static partial class defchararray
            {
                /// <summary>
                ///     For each element, return True if there are only numeric
                ///     characters in the element.<br></br>
                ///     Calls unicode.isnumeric element-wise.<br></br>
                ///     Numeric characters include digit characters, and all characters
                ///     that have the Unicode numeric value property, e.g.<br></br>
                ///     U+2155,
                ///     VULGAR FRACTION ONE FIFTH.
                /// </summary>
                /// <param name="a">
                ///     Input array.
                /// </param>
                /// <returns>
                ///     Array of booleans of same shape as a.
                /// </returns>
                public static NDarray isnumeric(NDarray a)
                {
                    if (Gpu.Available && Gpu.Use)
                    {
                        return new NDarray(cp.core.defchararray.isnumeric(a.CupyNDarray));
                    }
                    else
                    {
                        return new NDarray(np.core.defchararray.isnumeric(a.NumpyNDarray));
                    }
                }
            }
        }

        public static partial class core
        {
            public static partial class defchararray
            {
                /// <summary>
                ///     Returns true for each element if there are only whitespace
                ///     characters in the string and there is at least one character,
                ///     false otherwise.<br></br>
                ///     Calls str.isspace element-wise.<br></br>
                ///     For 8-bit strings, this method is locale-dependent.
                /// </summary>
                /// <returns>
                ///     Output array of bools
                /// </returns>
                public static NDarray isspace(params string[] a)
                {
                    if (Gpu.Available && Gpu.Use)
                    {
                        return new NDarray(cp.core.defchararray.isspace(a));
                    }
                    else
                    {
                        return new NDarray(np.core.defchararray.isspace(a));
                    }
                }
            }
        }

        public static partial class core
        {
            public static partial class defchararray
            {
                /// <summary>
                ///     Returns true for each element if the element is a titlecased
                ///     string and there is at least one character, false otherwise.<br></br>
                ///     Call str.istitle element-wise.<br></br>
                ///     For 8-bit strings, this method is locale-dependent.
                /// </summary>
                /// <returns>
                ///     Output array of bools
                /// </returns>
                public static NDarray istitle(params string[] a)
                {
                    if (Gpu.Available && Gpu.Use)
                    {
                        return new NDarray(cp.core.defchararray.istitle(a));
                    }
                    else
                    {
                        return new NDarray(np.core.defchararray.istitle(a));
                    }
                }
            }
        }

        public static partial class core
        {
            public static partial class defchararray
            {
                /// <summary>
                ///     Returns true for each element if all cased characters in the
                ///     string are uppercase and there is at least one character, false
                ///     otherwise.<br></br>
                ///     Call str.isupper element-wise.<br></br>
                ///     For 8-bit strings, this method is locale-dependent.
                /// </summary>
                /// <returns>
                ///     Output array of bools
                /// </returns>
                public static NDarray isupper(params string[] a)
                {
                    if (Gpu.Available && Gpu.Use)
                    {
                        return new NDarray(cp.core.defchararray.isupper(a));
                    }
                    else
                    {
                        return new NDarray(np.core.defchararray.isupper(a));
                    }
                }
            }
        }

        public static partial class core
        {
            public static partial class defchararray
            {
                /// <summary>
                ///     For each element in a, return the highest index in the string
                ///     where substring sub is found, such that sub is contained
                ///     within [start, end].<br></br>
                ///     Calls str.rfind element-wise.
                /// </summary>
                /// <param name="end">
                ///     Optional arguments start and end are interpreted as in
                ///     slice notation.
                /// </param>
                /// <param name="start">
                ///     Optional arguments start and end are interpreted as in
                ///     slice notation.
                /// </param>
                /// <returns>
                ///     Output array of ints.<br></br>
                ///     Return -1 on failure.
                /// </returns>
                public static NDarray rfind(string[] a, string sub, int? end = null, int? start = null)
                {
                    if (Gpu.Available && Gpu.Use)
                    {
                        return new NDarray(cp.core.defchararray.rfind(a, sub, end, start));
                    }
                    else
                    {
                        return new NDarray(np.core.defchararray.rfind(a, sub, end, start));
                    }
                }
            }
        }

        public static partial class core
        {
            public static partial class defchararray
            {
                /// <summary>
                ///     Like rfind, but raises ValueError when the substring sub is
                ///     not found.<br></br>
                ///     Calls str.rindex element-wise.
                /// </summary>
                /// <returns>
                ///     Output array of ints.
                /// </returns>
                public static NDarray rindex(string[] a, string sub, int? end = null, int? start = null)
                {
                    if (Gpu.Available && Gpu.Use)
                    {
                        return new NDarray(cp.core.defchararray.rindex(a, sub, end, start));
                    }
                    else
                    {
                        return new NDarray(np.core.defchararray.rindex(a, sub, end, start));
                    }
                }
            }
        }

        public static partial class core
        {
            public static partial class defchararray
            {
                /// <summary>
                ///     Returns a boolean array which is True where the string element
                ///     in a starts with prefix, otherwise False.<br></br>
                ///     Calls str.startswith element-wise.
                /// </summary>
                /// <param name="end">
                ///     With optional start, test beginning at that position.<br></br>
                ///     With
                ///     optional end, stop comparing at that position.
                /// </param>
                /// <param name="start">
                ///     With optional start, test beginning at that position.<br></br>
                ///     With
                ///     optional end, stop comparing at that position.
                /// </param>
                /// <returns>
                ///     Array of booleans
                /// </returns>
                public static NDarray startswith(string[] a, string prefix, int? end = null, int? start = null)
                {
                    if (Gpu.Available && Gpu.Use)
                    {
                        return new NDarray(cp.core.defchararray.startswith(a, prefix, end, start));
                    }
                    else
                    {
                        return new NDarray(np.core.defchararray.startswith(a, prefix, end, start));
                    }
                }
            }
        }

        public static partial class core
        {
            public static partial class defchararray
            {
                /// <summary>
                ///     Provides a convenient view on arrays of string and unicode values.<br></br>
                ///     Versus a regular Cupy array of type str or unicode, this
                ///     class adds the following functionality:
                ///     chararrays should be created using Cupy.char.array or
                ///     Cupy.char.asarray, rather than this constructor directly.<br></br>
                ///     This constructor creates the array, using buffer (with offset
                ///     and strides) if it is not None.<br></br>
                ///     If buffer is None, then
                ///     constructs a new array with strides in “C order”, unless both
                ///     len(shape) &gt;= 2 and order='Fortran', in which case strides
                ///     is in “Fortran order”.
                /// </summary>
                /// <param name="shape">
                ///     Shape of the array.
                /// </param>
                /// <param name="itemsize">
                ///     Length of each array element, in number of characters.<br></br>
                ///     Default is 1.
                /// </param>
                /// <param name="unicode">
                ///     Are the array elements of type unicode (True) or string (False).<br></br>
                ///     Default is False.
                /// </param>
                /// <param name="buffer">
                ///     Memory address of the start of the array data.<br></br>
                ///     Default is None,
                ///     in which case a new array is created.
                /// </param>
                /// <param name="offset">
                ///     Fixed stride displacement from the beginning of an axis?
                ///     Default is 0.<br></br>
                ///     Needs to be &gt;=0.
                /// </param>
                /// <param name="strides">
                ///     Strides for the array (see ndarray.strides for full description).<br></br>
                ///     Default is None.
                /// </param>
                /// <param name="order">
                ///     The order in which the array data is stored in memory: ‘C’ -&gt;
                ///     “row major” order (the default), ‘F’ -&gt; “column major”
                ///     (Fortran) order.
                /// </param>
                public static void chararray(Shape shape, int? itemsize = null, bool? unicode = null,
                    int? buffer = null, int? offset = null, int[] strides = null, string order = null)
                {
                    if (Gpu.Available && Gpu.Use)
                    {
                        cp.core.defchararray.chararray(shape.CupyShape, itemsize, unicode, buffer, offset, strides, order);
                    }
                    else
                    {
                        np.core.defchararray.chararray(shape.NumpyShape, itemsize, unicode, buffer, offset, strides, order);
                    }
                }
            }
        }
    }
}
