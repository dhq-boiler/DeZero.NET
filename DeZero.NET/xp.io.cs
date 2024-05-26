﻿using Cupy;
using Numpy;
using Python.Runtime;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeZero.NET
{
    public static partial class xp
    {
        /// <summary>
        ///     Load arrays or pickled objects from .npy, .npz or pickled files.<br></br>
        ///     Notes
        /// </summary>
        /// <param name="file">
        ///     The file to read.<br></br>
        ///     File-like objects must support the
        ///     seek() and read() methods.<br></br>
        ///     Pickled files require that the
        ///     file-like object support the readline() method as well.
        /// </param>
        /// <param name="mmap_mode">
        ///     If not None, then memory-map the file, using the given mode (see
        ///     Cupy.memmap for a detailed description of the modes).<br></br>
        ///     A
        ///     memory-mapped array is kept on disk.<br></br>
        ///     However, it can be accessed
        ///     and sliced like any ndarray.<br></br>
        ///     Memory mapping is especially useful
        ///     for accessing small fragments of large files without reading the
        ///     entire file into memory.
        /// </param>
        /// <param name="allow_pickle">
        ///     Allow loading pickled object arrays stored in npy files.<br></br>
        ///     Reasons for
        ///     disallowing pickles include security, as loading pickled data can
        ///     execute arbitrary code.<br></br>
        ///     If pickles are disallowed, loading object
        ///     arrays will fail.<br></br>
        ///     Default: True
        /// </param>
        /// <param name="fix_imports">
        ///     Only useful when loading Python 2 generated pickled files on Python 3,
        ///     which includes npy/npz files containing object arrays.<br></br>
        ///     If fix_imports
        ///     is True, pickle will try to map the old Python 2 names to the new names
        ///     used in Python 3.
        /// </param>
        /// <param name="encoding">
        ///     What encoding to use when reading Python 2 strings.<br></br>
        ///     Only useful when
        ///     loading Python 2 generated pickled files in Python 3, which includes
        ///     npy/npz files containing object arrays.<br></br>
        ///     Values other than ‘latin1’,
        ///     ‘ASCII’, and ‘bytes’ are not allowed, as they can corrupt numerical
        ///     data.<br></br>
        ///     Default: ‘ASCII’
        /// </param>
        /// <returns>
        ///     Data stored in the file.<br></br>
        ///     For .npz files, the returned instance
        ///     of NpzFile class must be closed to avoid leaking file descriptors.
        /// </returns>
        public static NDarray load(string file, MemMapMode mmap_mode = null, bool? allow_pickle = false,
            bool? fix_imports = true, string encoding = "ASCII")
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.load(file, mmap_mode.CupyMemMapMode, allow_pickle, fix_imports, encoding));
            }
            else
            {
                return new NDarray(np.load(file, mmap_mode.NumpyMemMapMode, allow_pickle, fix_imports, encoding));
            }
        }

        /// <summary>
        ///     Save an array to a text file.<br></br>
        ///     Notes
        ///     Further explanation of the fmt parameter
        ///     (%[flag]width[.precision]specifier):
        ///     This explanation of fmt is not complete, for an exhaustive
        ///     specification see [1].<br></br>
        ///     References
        /// </summary>
        /// <param name="fname">
        ///     If the filename ends in .gz, the file is automatically saved in
        ///     compressed gzip format.<br></br>
        ///     loadtxt understands gzipped files
        ///     transparently.
        /// </param>
        /// <param name="X">
        ///     Data to be saved to a text file.
        /// </param>
        /// <param name="fmt">
        ///     A single format (%10.5f), a sequence of formats, or a
        ///     multi-format string, e.g.<br></br>
        ///     ‘Iteration %d – %10.5f’, in which
        ///     case delimiter is ignored.<br></br>
        ///     For complex X, the legal options
        ///     for fmt are:
        /// </param>
        /// <param name="delimiter">
        ///     String or character separating columns.
        /// </param>
        /// <param name="newline">
        ///     String or character separating lines.
        /// </param>
        /// <param name="header">
        ///     String that will be written at the beginning of the file.
        /// </param>
        /// <param name="footer">
        ///     String that will be written at the end of the file.
        /// </param>
        /// <param name="comments">
        ///     String that will be prepended to the header and footer strings,
        ///     to mark them as comments.<br></br>
        ///     Default: ‘# ‘,  as expected by e.g.<br></br>
        ///     Cupy.loadtxt.
        /// </param>
        /// <param name="encoding">
        ///     Encoding used to encode the outputfile.<br></br>
        ///     Does not apply to output
        ///     streams.<br></br>
        ///     If the encoding is something other than ‘bytes’ or ‘latin1’
        ///     you will not be able to load the file in Cupy versions &lt; 1.14. Default
        ///     is ‘latin1’.
        /// </param>
        public static void savetxt(string fname, NDarray X, string[] fmt = null, string delimiter = " ",
            string newline = "\n", string header = "", string footer = "", string comments = null,
            string encoding = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                cp.savetxt(fname, X.CupyNDarray, fmt, delimiter, newline, header, footer, comments, encoding);
            }
            else
            {
                np.savetxt(fname, X.NumpyNDarray, fmt, delimiter, newline, header, footer, comments, encoding);
            }
        }

        /*
        /// <summary>
        ///	Load data from a text file, with missing values handled as specified.<br></br>
        ///	
        ///	Each line past the first skip_header lines is split at the delimiter
        ///	character, and characters following the comments character are discarded.<br></br>
        ///	
        ///	Notes
        ///	
        ///	References
        /// </summary>
        /// <param name="fname">
        ///	File, filename, list, or generator to read.<br></br>
        ///	If the filename
        ///	extension is gz or bz2, the file is first decompressed.<br></br>
        ///	Note
        ///	that generators must return byte strings in Python 3k.<br></br>
        ///	The strings
        ///	in a list or produced by a generator are treated as lines.
        /// </param>
        /// <param name="dtype">
        ///	Data type of the resulting array.<br></br>
        ///	
        ///	If None, the dtypes will be determined by the contents of each
        ///	column, individually.
        /// </param>
        /// <param name="comments">
        ///	The character used to indicate the start of a comment.<br></br>
        ///	
        ///	All the characters occurring on a line after a comment are discarded
        /// </param>
        /// <param name="delimiter">
        ///	The string used to separate values.<br></br>
        ///	By default, any consecutive
        ///	whitespaces act as delimiter.<br></br>
        ///	An integer or sequence of integers
        ///	can also be provided as width(s) of each field.
        /// </param>
        /// <param name="skiprows">
        ///	skiprows was removed in Cupy 1.10. Please use skip_header instead.
        /// </param>
        /// <param name="skip_header">
        ///	The number of lines to skip at the beginning of the file.
        /// </param>
        /// <param name="skip_footer">
        ///	The number of lines to skip at the end of the file.
        /// </param>
        /// <param name="converters">
        ///	The set of functions that convert the data of a column to a value.<br></br>
        ///	
        ///	The converters can also be used to provide a default value
        ///	for missing data: converters = {3: lambda s: float(s or 0)}.
        /// </param>
        /// <param name="missing">
        ///	missing was removed in Cupy 1.10. Please use missing_values
        ///	instead.
        /// </param>
        /// <param name="missing_values">
        ///	The set of strings corresponding to missing data.
        /// </param>
        /// <param name="filling_values">
        ///	The set of values to be used as default when the data are missing.
        /// </param>
        /// <param name="usecols">
        ///	Which columns to read, with 0 being the first.<br></br>
        ///	For example,
        ///	usecols = (1, 4, 5) will extract the 2nd, 5th and 6th columns.
        /// </param>
        /// <param name="names">
        ///	If names is True, the field names are read from the first line after
        ///	the first skip_header lines.<br></br>
        ///	This line can optionally be proceeded
        ///	by a comment delimiter.<br></br>
        ///	If names is a sequence or a single-string of
        ///	comma-separated names, the names will be used to define the field names
        ///	in a structured dtype.<br></br>
        ///	If names is None, the names of the dtype
        ///	fields will be used, if any.
        /// </param>
        /// <param name="excludelist">
        ///	A list of names to exclude.<br></br>
        ///	This list is appended to the default list
        ///	[‘return’,’file’,’print’].<br></br>
        ///	Excluded names are appended an underscore:
        ///	for example, file would become file_.
        /// </param>
        /// <param name="deletechars">
        ///	A string combining invalid characters that must be deleted from the
        ///	names.
        /// </param>
        /// <param name="defaultfmt">
        ///	A format used to define default field names, such as “f%i” or “f_%02i”.
        /// </param>
        /// <param name="autostrip">
        ///	Whether to automatically strip white spaces from the variables.
        /// </param>
        /// <param name="replace_space">
        ///	Character(s) used in replacement of white spaces in the variables
        ///	names.<br></br>
        ///	By default, use a ‘_’.
        /// </param>
        /// <param name="case_sensitive">
        ///	If True, field names are case sensitive.<br></br>
        ///	
        ///	If False or ‘upper’, field names are converted to upper case.<br></br>
        ///	
        ///	If ‘lower’, field names are converted to lower case.
        /// </param>
        /// <param name="unpack">
        ///	If True, the returned array is transposed, so that arguments may be
        ///	unpacked using x, y, z = loadtxt(...)
        /// </param>
        /// <param name="usemask">
        ///	If True, return a masked array.<br></br>
        ///	
        ///	If False, return a regular array.
        /// </param>
        /// <param name="loose">
        ///	If True, do not raise errors for invalid values.
        /// </param>
        /// <param name="invalid_raise">
        ///	If True, an exception is raised if an inconsistency is detected in the
        ///	number of columns.<br></br>
        ///	
        ///	If False, a warning is emitted and the offending lines are skipped.
        /// </param>
        /// <param name="max_rows">
        ///	The maximum number of rows to read.<br></br>
        ///	Must not be used with skip_footer
        ///	at the same time.<br></br>
        ///	If given, the value must be at least 1.<br></br>
        ///	Default is
        ///	to read the entire file.
        /// </param>
        /// <param name="encoding">
        ///	Encoding used to decode the inputfile.<br></br>
        ///	Does not apply when fname is
        ///	a file object.<br></br>
        ///	The special value ‘bytes’ enables backward compatibility
        ///	workarounds that ensure that you receive byte arrays when possible
        ///	and passes latin1 encoded strings to converters.<br></br>
        ///	Override this value to
        ///	receive unicode arrays and pass strings as input to converters.<br></br>
        ///	If set
        ///	to None the system default is used.<br></br>
        ///	The default value is ‘bytes’.
        /// </param>
        /// <returns>
        ///	Data read from the text file.<br></br>
        ///	 If usemask is True, this is a
        ///	masked array.
        /// </returns>
        public static NDarray genfromtxt(string fname, Dtype dtype = null, string comments = null, string delimiter = null, int? skiprows = null, int? skip_header = 0, int? skip_footer = 0, variable converters = null, variable missing = null, variable missing_values = null, variable filling_values = null, sequence usecols = null, {None names = null, sequence excludelist = null, string deletechars = null, string defaultfmt = "f%i", bool? autostrip = false, string replace_space = "_", {True case_sensitive = true, bool? unpack = null, bool? usemask = false, bool? loose = true, bool? invalid_raise = true, int? max_rows = null, string encoding = "bytes")
        {
            //auto-generated code, do not change
            var __self__=self;
            var pyargs=ToTuple(new object[]
            {
                fname,
            });
            var kwargs=new PyDict();
            if (dtype!=null) kwargs["dtype"]=ToPython(dtype);
            if (comments!=null) kwargs["comments"]=ToPython(comments);
            if (delimiter!=null) kwargs["delimiter"]=ToPython(delimiter);
            if (skiprows!=null) kwargs["skiprows"]=ToPython(skiprows);
            if (skip_header!=0) kwargs["skip_header"]=ToPython(skip_header);
            if (skip_footer!=0) kwargs["skip_footer"]=ToPython(skip_footer);
            if (converters!=null) kwargs["converters"]=ToPython(converters);
            if (missing!=null) kwargs["missing"]=ToPython(missing);
            if (missing_values!=null) kwargs["missing_values"]=ToPython(missing_values);
            if (filling_values!=null) kwargs["filling_values"]=ToPython(filling_values);
            if (usecols!=null) kwargs["usecols"]=ToPython(usecols);
            if (names!=null) kwargs["names"]=ToPython(names);
            if (excludelist!=null) kwargs["excludelist"]=ToPython(excludelist);
            if (deletechars!=null) kwargs["deletechars"]=ToPython(deletechars);
            if (defaultfmt!="f%i") kwargs["defaultfmt"]=ToPython(defaultfmt);
            if (autostrip!=false) kwargs["autostrip"]=ToPython(autostrip);
            if (replace_space!="_") kwargs["replace_space"]=ToPython(replace_space);
            if (case_sensitive!=true) kwargs["case_sensitive"]=ToPython(case_sensitive);
            if (unpack!=null) kwargs["unpack"]=ToPython(unpack);
            if (usemask!=false) kwargs["usemask"]=ToPython(usemask);
            if (loose!=true) kwargs["loose"]=ToPython(loose);
            if (invalid_raise!=true) kwargs["invalid_raise"]=ToPython(invalid_raise);
            if (max_rows!=null) kwargs["max_rows"]=ToPython(max_rows);
            if (encoding!="bytes") kwargs["encoding"]=ToPython(encoding);
            dynamic py = __self__.InvokeMethod("genfromtxt", pyargs, kwargs);
            return ToCsharp<NDarray>(py);
        }
        */

        /// <summary>
        ///     Construct an array from a text file, using regular expression parsing.<br></br>
        ///     The returned array is always a structured array, and is constructed from
        ///     all matches of the regular expression in the file.<br></br>
        ///     Groups in the regular
        ///     expression are converted to fields of the structured array.<br></br>
        ///     Notes
        ///     Dtypes for structured arrays can be specified in several forms, but all
        ///     forms specify at least the data type and field name.<br></br>
        ///     For details see
        ///     doc.structured_arrays.
        /// </summary>
        /// <param name="file">
        ///     File name or file object to read.
        /// </param>
        /// <param name="regexp">
        ///     Regular expression used to parse the file.<br></br>
        ///     Groups in the regular expression correspond to fields in the dtype.
        /// </param>
        /// <param name="dtype">
        ///     Dtype for the structured array.
        /// </param>
        /// <param name="encoding">
        ///     Encoding used to decode the inputfile.<br></br>
        ///     Does not apply to input streams.
        /// </param>
        /// <returns>
        ///     The output array, containing the part of the content of file that
        ///     was matched by regexp.<br></br>
        ///     output is always a structured array.
        /// </returns>
        public static NDarray fromregex(string file, string regexp, Dtype dtype, string encoding = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.fromregex(file, regexp, dtype.CupyDtype, encoding));
            }
            else
            {
                return new NDarray(np.fromregex(file, regexp, dtype.NumpyDtype, encoding));
            }
        }

        /// <summary>
        ///     Write array to a file as text or binary (default).<br></br>
        ///     Data is always written in ‘C’ order, independent of the order of a.<br></br>
        ///     The data produced by this method can be recovered using the function
        ///     fromfile().<br></br>
        ///     Notes
        ///     This is a convenience function for quick storage of array data.<br></br>
        ///     Information on endianness and precision is lost, so this method is not a
        ///     good choice for files intended to archive data or transport data between
        ///     machines with different endianness.<br></br>
        ///     Some of these problems can be overcome
        ///     by outputting the data as text files, at the expense of speed and file
        ///     size.<br></br>
        ///     When fid is a file object, array contents are directly written to the
        ///     file, bypassing the file object’s write method.<br></br>
        ///     As a result, tofile
        ///     cannot be used with files objects supporting compression (e.g., GzipFile)
        ///     or file-like objects that do not support fileno() (e.g., BytesIO).
        /// </summary>
        /// <param name="fid">
        ///     An open file object, or a string containing a filename.
        /// </param>
        /// <param name="sep">
        ///     Separator between array items for text output.<br></br>
        ///     If “” (empty), a binary file is written, equivalent to
        ///     file.write(a.tobytes()).
        /// </param>
        /// <param name="format">
        ///     Format string for text file output.<br></br>
        ///     Each entry in the array is formatted to text by first converting
        ///     it to the closest Python type, and then using “format” % item.
        /// </param>
        public static void tofile(string fid, string sep, string format)
        {
            if (Gpu.Available && Gpu.Use)
            {
                cp.tofile(fid, sep, format);
            }
            else
            {
                np.tofile(fid, sep, format);
            }
        }

        /*
        /// <summary>
        ///	Return the array as a (possibly nested) list.<br></br>
        ///	
        ///	Return a copy of the array data as a (nested) Python list.<br></br>
        ///	
        ///	Data items are converted to the nearest compatible Python type.<br></br>
        ///	
        ///	Notes
        ///	
        ///	The array may be recreated, a = cp.array(a.tolist()).
        /// </summary>
        /// <returns>
        ///	The possibly nested list of array elements.
        /// </returns>
        public static List<T> tolist<T>()
        {
            //auto-generated code, do not change
            var __self__=self;
            dynamic py = __self__.InvokeMethod("tolist");
            return ToCsharp<List<T>>(py);
        }
        */

        /*
        /// <summary>
        ///	Return a string representation of an array.<br></br>
        ///	
        ///	Notes
        ///	
        ///	If a formatter is specified for a certain type, the precision keyword is
        ///	ignored for that type.<br></br>
        ///	
        ///	This is a very flexible function; array_repr and array_str are using
        ///	array2string internally so keywords with the same name should work
        ///	identically in all three functions.
        /// </summary>
        /// <param name="a">
        ///	Input array.
        /// </param>
        /// <param name="max_line_width">
        ///	The maximum number of columns the string should span.<br></br>
        ///	Newline
        ///	characters splits the string appropriately after array elements.
        /// </param>
        /// <param name="precision">
        ///	Floating point precision.<br></br>
        ///	Default is the current printing
        ///	precision (usually 8), which can be altered using set_printoptions.
        /// </param>
        /// <param name="suppress_small">
        ///	Represent very small numbers as zero.<br></br>
        ///	A number is “very small” if it
        ///	is smaller than the current printing precision.
        /// </param>
        /// <param name="separator">
        ///	Inserted between elements.
        /// </param>
        /// <param name="suffix">
        ///	The length of the prefix and suffix strings are used to respectively
        ///	align and wrap the output.<br></br>
        ///	An array is typically printed as:
        ///	
        ///	The output is left-padded by the length of the prefix string, and
        ///	wrapping is forced at the column max_line_width - len(suffix).<br></br>
        ///	
        ///	It should be noted that the content of prefix and suffix strings are
        ///	not included in the output.
        /// </param>
        /// <param name="formatter">
        ///	If not None, the keys should indicate the type(s) that the respective
        ///	formatting function applies to.<br></br>
        ///	Callables should return a string.<br></br>
        ///	
        ///	Types that are not specified (by their corresponding keys) are handled
        ///	by the default formatters.<br></br>
        ///	Individual types for which a formatter
        ///	can be set are:
        ///	
        ///	Other keys that can be used to set a group of types at once are:
        /// </param>
        /// <param name="threshold">
        ///	Total number of array elements which trigger summarization
        ///	rather than full repr.
        /// </param>
        /// <param name="edgeitems">
        ///	Number of array items in summary at beginning and end of
        ///	each dimension.
        /// </param>
        /// <param name="sign">
        ///	Controls printing of the sign of floating-point types.<br></br>
        ///	If ‘+’, always
        ///	print the sign of positive values.<br></br>
        ///	If ‘ ‘, always prints a space
        ///	(whitespace character) in the sign position of positive values.<br></br>
        ///	If
        ///	‘-‘, omit the sign character of positive values.
        /// </param>
        /// <param name="floatmode">
        ///	Controls the interpretation of the precision option for
        ///	floating-point types.<br></br>
        ///	Can take the following values:
        /// </param>
        /// <param name="legacy">
        ///	If set to the string ‘1.13’ enables 1.13 legacy printing mode.<br></br>
        ///	This
        ///	approximates Cupy 1.13 print output by including a space in the sign
        ///	position of floats and different behavior for 0d arrays.<br></br>
        ///	If set to
        ///	False, disables legacy mode.<br></br>
        ///	Unrecognized strings will be ignored
        ///	with a warning for forward compatibility.
        /// </param>
        /// <returns>
        ///	String representation of the array.
        /// </returns>
        public static string array2string(this NDarray a, int? max_line_width = null, int? precision = null, bool? suppress_small = null, string separator = " ", string prefix = "", string suffix = "", dict of callables formatter = null, int? threshold = null, int? edgeitems = null, string sign = null, string floatmode = null, string or False legacy = null)
        {
            //auto-generated code, do not change
            var __self__=self;
            var pyargs=ToTuple(new object[]
            {
                a,
            });
            var kwargs=new PyDict();
            if (max_line_width!=null) kwargs["max_line_width"]=ToPython(max_line_width);
            if (precision!=null) kwargs["precision"]=ToPython(precision);
            if (suppress_small!=null) kwargs["suppress_small"]=ToPython(suppress_small);
            if (separator!=" ") kwargs["separator"]=ToPython(separator);
            if (prefix!="") kwargs["prefix"]=ToPython(prefix);
            if (suffix!="") kwargs["suffix"]=ToPython(suffix);
            if (formatter!=null) kwargs["formatter"]=ToPython(formatter);
            if (threshold!=null) kwargs["threshold"]=ToPython(threshold);
            if (edgeitems!=null) kwargs["edgeitems"]=ToPython(edgeitems);
            if (sign!=null) kwargs["sign"]=ToPython(sign);
            if (floatmode!=null) kwargs["floatmode"]=ToPython(floatmode);
            if (legacy!=null) kwargs["legacy"]=ToPython(legacy);
            dynamic py = __self__.InvokeMethod("array2string", pyargs, kwargs);
            return ToCsharp<string>(py);
        }
        */

        /// <summary>
        ///     Return the string representation of an array.
        /// </summary>
        /// <param name="arr">
        ///     Input array.
        /// </param>
        /// <param name="max_line_width">
        ///     The maximum number of columns the string should span.<br></br>
        ///     Newline
        ///     characters split the string appropriately after array elements.
        /// </param>
        /// <param name="precision">
        ///     Floating point precision.<br></br>
        ///     Default is the current printing precision
        ///     (usually 8), which can be altered using set_printoptions.
        /// </param>
        /// <param name="suppress_small">
        ///     Represent very small numbers as zero, default is False.<br></br>
        ///     Very small
        ///     is defined by precision, if the precision is 8 then
        ///     numbers smaller than 5e-9 are represented as zero.
        /// </param>
        /// <returns>
        ///     The string representation of an array.
        /// </returns>
        public static string array_repr(this NDarray arr, int? max_line_width = null, int? precision = null,
            bool? suppress_small = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return cp.array_repr(arr.CupyNDarray, max_line_width, precision, suppress_small);
            }
            else
            {
                return np.array_repr(arr.NumpyNDarray, max_line_width, precision, suppress_small);
            }
        }

        /// <summary>
        ///     Return a string representation of the data in an array.<br></br>
        ///     The data in the array is returned as a single string.<br></br>
        ///     This function is
        ///     similar to array_repr, the difference being that array_repr also
        ///     returns information on the kind of array and its data type.
        /// </summary>
        /// <param name="a">
        ///     Input array.
        /// </param>
        /// <param name="max_line_width">
        ///     Inserts newlines if text is longer than max_line_width.<br></br>
        ///     The
        ///     default is, indirectly, 75.
        /// </param>
        /// <param name="precision">
        ///     Floating point precision.<br></br>
        ///     Default is the current printing precision
        ///     (usually 8), which can be altered using set_printoptions.
        /// </param>
        /// <param name="suppress_small">
        ///     Represent numbers “very close” to zero as zero; default is False.<br></br>
        ///     Very close is defined by precision: if the precision is 8, e.g.,
        ///     numbers smaller (in absolute value) than 5e-9 are represented as
        ///     zero.
        /// </param>
        public static void array_str(this NDarray a, int? max_line_width = null, int? precision = null,
            bool? suppress_small = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                cp.array_str(a.CupyNDarray, max_line_width, precision, suppress_small);
            }
            else
            {
                np.array_str(a.NumpyNDarray, max_line_width, precision, suppress_small);
            }
        }

        /*
        /// <summary>
        ///	Format a floating-point scalar as a decimal string in positional notation.<br></br>
        ///	
        ///	Provides control over rounding, trimming and padding.<br></br>
        ///	 Uses and assumes
        ///	IEEE unbiased rounding.<br></br>
        ///	 Uses the “Dragon4” algorithm.
        /// </summary>
        /// <param name="x">
        ///	Value to format.
        /// </param>
        /// <param name="precision">
        ///	Maximum number of digits to print.<br></br>
        ///	May be None if unique is
        ///	True, but must be an integer if unique is False.
        /// </param>
        /// <param name="unique">
        ///	If True, use a digit-generation strategy which gives the shortest
        ///	representation which uniquely identifies the floating-point number from
        ///	other values of the same type, by judicious rounding.<br></br>
        ///	If precision
        ///	was omitted, print out all necessary digits, otherwise digit generation
        ///	is cut off after precision digits and the remaining value is rounded.<br></br>
        ///	
        ///	If False, digits are generated as if printing an infinite-precision
        ///	value and stopping after precision digits, rounding the remaining
        ///	value.
        /// </param>
        /// <param name="fractional">
        ///	If True, the cutoff of precision digits refers to the total number
        ///	of digits after the decimal point, including leading zeros.<br></br>
        ///	
        ///	If False, precision refers to the total number of significant
        ///	digits, before or after the decimal point, ignoring leading zeros.
        /// </param>
        /// <param name="trim">
        ///	Controls post-processing trimming of trailing digits, as follows:
        /// </param>
        /// <param name="sign">
        ///	Whether to show the sign for positive values.
        /// </param>
        /// <param name="pad_left">
        ///	Pad the left side of the string with whitespace until at least that
        ///	many characters are to the left of the decimal point.
        /// </param>
        /// <param name="pad_right">
        ///	Pad the right side of the string with whitespace until at least that
        ///	many characters are to the right of the decimal point.
        /// </param>
        /// <returns>
        ///	The string representation of the floating point value
        /// </returns>
        public static string format_float_positional(python float or Cupy floating scalar x, non-negative integer or None precision = null, bool? unique = true, bool? fractional = true, string trim = "k", bool? sign = false, non-negative integer pad_left = null, non-negative integer pad_right = null)
        {
            //auto-generated code, do not change
            var __self__=self;
            var pyargs=ToTuple(new object[]
            {
                x,
            });
            var kwargs=new PyDict();
            if (precision!=null) kwargs["precision"]=ToPython(precision);
            if (unique!=true) kwargs["unique"]=ToPython(unique);
            if (fractional!=true) kwargs["fractional"]=ToPython(fractional);
            if (trim!="k") kwargs["trim"]=ToPython(trim);
            if (sign!=false) kwargs["sign"]=ToPython(sign);
            if (pad_left!=null) kwargs["pad_left"]=ToPython(pad_left);
            if (pad_right!=null) kwargs["pad_right"]=ToPython(pad_right);
            dynamic py = __self__.InvokeMethod("format_float_positional", pyargs, kwargs);
            return ToCsharp<string>(py);
        }
        */

        /*
        /// <summary>
        ///	Format a floating-point scalar as a decimal string in scientific notation.<br></br>
        ///	
        ///	Provides control over rounding, trimming and padding.<br></br>
        ///	 Uses and assumes
        ///	IEEE unbiased rounding.<br></br>
        ///	 Uses the “Dragon4” algorithm.
        /// </summary>
        /// <param name="x">
        ///	Value to format.
        /// </param>
        /// <param name="precision">
        ///	Maximum number of digits to print.<br></br>
        ///	May be None if unique is
        ///	True, but must be an integer if unique is False.
        /// </param>
        /// <param name="unique">
        ///	If True, use a digit-generation strategy which gives the shortest
        ///	representation which uniquely identifies the floating-point number from
        ///	other values of the same type, by judicious rounding.<br></br>
        ///	If precision
        ///	was omitted, print all necessary digits, otherwise digit generation is
        ///	cut off after precision digits and the remaining value is rounded.<br></br>
        ///	
        ///	If False, digits are generated as if printing an infinite-precision
        ///	value and stopping after precision digits, rounding the remaining
        ///	value.
        /// </param>
        /// <param name="trim">
        ///	Controls post-processing trimming of trailing digits, as follows:
        /// </param>
        /// <param name="sign">
        ///	Whether to show the sign for positive values.
        /// </param>
        /// <param name="pad_left">
        ///	Pad the left side of the string with whitespace until at least that
        ///	many characters are to the left of the decimal point.
        /// </param>
        /// <param name="exp_digits">
        ///	Pad the exponent with zeros until it contains at least this many digits.<br></br>
        ///	
        ///	If omitted, the exponent will be at least 2 digits.
        /// </param>
        /// <returns>
        ///	The string representation of the floating point value
        /// </returns>
        public static string format_float_scientific(python float or Cupy floating scalar x, non-negative integer or None precision = null, bool? unique = true, string trim = "k", bool? sign = false, non-negative integer pad_left = null, non-negative integer exp_digits = null)
        {
            //auto-generated code, do not change
            var __self__=self;
            var pyargs=ToTuple(new object[]
            {
                x,
            });
            var kwargs=new PyDict();
            if (precision!=null) kwargs["precision"]=ToPython(precision);
            if (unique!=true) kwargs["unique"]=ToPython(unique);
            if (trim!="k") kwargs["trim"]=ToPython(trim);
            if (sign!=false) kwargs["sign"]=ToPython(sign);
            if (pad_left!=null) kwargs["pad_left"]=ToPython(pad_left);
            if (exp_digits!=null) kwargs["exp_digits"]=ToPython(exp_digits);
            dynamic py = __self__.InvokeMethod("format_float_scientific", pyargs, kwargs);
            return ToCsharp<string>(py);
        }
        */

        /// <summary>
        ///     Create a memory-map to an array stored in a binary file on disk.<br></br>
        ///     Memory-mapped files are used for accessing small segments of large files
        ///     on disk, without reading the entire file into memory.<br></br>
        ///     Cupy’s
        ///     memmap’s are array-like objects.<br></br>
        ///     This differs from Python’s mmap
        ///     module, which uses file-like objects.<br></br>
        ///     This subclass of ndarray has some unpleasant interactions with
        ///     some operations, because it doesn’t quite fit properly as a subclass.<br></br>
        ///     An alternative to using this subclass is to create the mmap
        ///     object yourself, then create an ndarray with ndarray.__new__ directly,
        ///     passing the object created in its ‘buffer=’ parameter.<br></br>
        ///     This class may at some point be turned into a factory function
        ///     which returns a view into an mmap buffer.<br></br>
        ///     Delete the memmap instance to close the memmap file.<br></br>
        ///     Notes
        ///     The memmap object can be used anywhere an ndarray is accepted.<br></br>
        ///     Given a memmap fp, isinstance(fp, Cupy.ndarray) returns
        ///     True.<br></br>
        ///     Memory-mapped files cannot be larger than 2GB on 32-bit systems.<br></br>
        ///     When a memmap causes a file to be created or extended beyond its
        ///     current size in the filesystem, the contents of the new part are
        ///     unspecified.<br></br>
        ///     On systems with POSIX filesystem semantics, the extended
        ///     part will be filled with zero bytes.
        /// </summary>
        /// <param name="filename">
        ///     The file name or file object to be used as the array data buffer.
        /// </param>
        /// <param name="dtype">
        ///     The data-type used to interpret the file contents.<br></br>
        ///     Default is uint8.
        /// </param>
        /// <param name="mode">
        ///     The file is opened in this mode:
        ///     Default is ‘r+’.
        /// </param>
        /// <param name="offset">
        ///     In the file, array data starts at this offset.<br></br>
        ///     Since offset is
        ///     measured in bytes, it should normally be a multiple of the byte-size
        ///     of dtype.<br></br>
        ///     When mode != 'r', even positive offsets beyond end of
        ///     file are valid; The file will be extended to accommodate the
        ///     additional data.<br></br>
        ///     By default, memmap will start at the beginning of
        ///     the file, even if filename is a file pointer fp and
        ///     fp.tell() != 0.
        /// </param>
        /// <param name="shape">
        ///     The desired shape of the array.<br></br>
        ///     If mode == 'r' and the number
        ///     of remaining bytes after offset is not a multiple of the byte-size
        ///     of dtype, you must specify shape.<br></br>
        ///     By default, the returned array
        ///     will be 1-D with the number of elements determined by file size
        ///     and data-type.
        /// </param>
        /// <param name="order">
        ///     Specify the order of the ndarray memory layout:
        ///     row-major, C-style or column-major,
        ///     Fortran-style.<br></br>
        ///     This only has an effect if the shape is
        ///     greater than 1-D.<br></br>
        ///     The default order is ‘C’.
        /// </param>
        public static void memmap(string filename, Dtype dtype = null, string mode = null, int? offset = null,
            Shape shape = null, string order = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                cp.memmap(filename, dtype?.CupyDtype, mode, offset, shape?.CupyShape, order);
            }
            else
            {
                np.memmap(filename, dtype?.NumpyDtype, mode, offset, shape?.NumpyShape, order);
            }
        }

        /*
        /// <summary>
        ///	SetKeywordArg printing options.<br></br>
        ///	
        ///	These options determine the way floating point numbers, arrays and
        ///	other Cupy objects are displayed.<br></br>
        ///	
        ///	Notes
        ///	
        ///	formatter is always reset with a call to set_printoptions.
        /// </summary>
        /// <param name="precision">
        ///	Number of digits of precision for floating point output (default 8).<br></br>
        ///	
        ///	May be None if floatmode is not fixed, to print as many digits as
        ///	necessary to uniquely specify the value.
        /// </param>
        /// <param name="threshold">
        ///	Total number of array elements which trigger summarization
        ///	rather than full repr (default 1000).
        /// </param>
        /// <param name="edgeitems">
        ///	Number of array items in summary at beginning and end of
        ///	each dimension (default 3).
        /// </param>
        /// <param name="linewidth">
        ///	The number of characters per line for the purpose of inserting
        ///	line breaks (default 75).
        /// </param>
        /// <param name="suppress">
        ///	If True, always print floating point numbers using fixed point
        ///	notation, in which case numbers equal to zero in the current precision
        ///	will print as zero.<br></br>
        ///	If False, then scientific notation is used when
        ///	absolute value of the smallest number is &lt; 1e-4 or the ratio of the
        ///	maximum absolute value to the minimum is &gt; 1e3. The default is False.
        /// </param>
        /// <param name="nanstr">
        ///	String representation of floating point not-a-number (default nan).
        /// </param>
        /// <param name="infstr">
        ///	String representation of floating point infinity (default inf).
        /// </param>
        /// <param name="sign">
        ///	Controls printing of the sign of floating-point types.<br></br>
        ///	If ‘+’, always
        ///	print the sign of positive values.<br></br>
        ///	If ‘ ‘, always prints a space
        ///	(whitespace character) in the sign position of positive values.<br></br>
        ///	If
        ///	‘-‘, omit the sign character of positive values.<br></br>
        ///	(default ‘-‘)
        /// </param>
        /// <param name="formatter">
        ///	If not None, the keys should indicate the type(s) that the respective
        ///	formatting function applies to.<br></br>
        ///	Callables should return a string.<br></br>
        ///	
        ///	Types that are not specified (by their corresponding keys) are handled
        ///	by the default formatters.<br></br>
        ///	Individual types for which a formatter
        ///	can be set are:
        ///	
        ///	Other keys that can be used to set a group of types at once are:
        /// </param>
        /// <param name="floatmode">
        ///	Controls the interpretation of the precision option for
        ///	floating-point types.<br></br>
        ///	Can take the following values:
        /// </param>
        /// <param name="legacy">
        ///	If set to the string ‘1.13’ enables 1.13 legacy printing mode.<br></br>
        ///	This
        ///	approximates Cupy 1.13 print output by including a space in the sign
        ///	position of floats and different behavior for 0d arrays.<br></br>
        ///	If set to
        ///	False, disables legacy mode.<br></br>
        ///	Unrecognized strings will be ignored
        ///	with a warning for forward compatibility.
        /// </param>
        public static void set_printoptions(int? precision = null, int? threshold = null, int? edgeitems = null, int? linewidth = null, bool? suppress = null, string nanstr = null, string infstr = null, string sign = null, dict of callables formatter = null, string floatmode = null, string or False legacy = null)
        {
            //auto-generated code, do not change
            var __self__=self;
            var pyargs=ToTuple(new object[]
            {
            });
            var kwargs=new PyDict();
            if (precision!=null) kwargs["precision"]=ToPython(precision);
            if (threshold!=null) kwargs["threshold"]=ToPython(threshold);
            if (edgeitems!=null) kwargs["edgeitems"]=ToPython(edgeitems);
            if (linewidth!=null) kwargs["linewidth"]=ToPython(linewidth);
            if (suppress!=null) kwargs["suppress"]=ToPython(suppress);
            if (nanstr!=null) kwargs["nanstr"]=ToPython(nanstr);
            if (infstr!=null) kwargs["infstr"]=ToPython(infstr);
            if (sign!=null) kwargs["sign"]=ToPython(sign);
            if (formatter!=null) kwargs["formatter"]=ToPython(formatter);
            if (floatmode!=null) kwargs["floatmode"]=ToPython(floatmode);
            if (legacy!=null) kwargs["legacy"]=ToPython(legacy);
            dynamic py = __self__.InvokeMethod("set_printoptions", pyargs, kwargs);
        }
        */

        /// <summary>
        ///     Return the current print options.
        /// </summary>
        /// <param name="print_opts">
        ///     KeywordArgs of current print options with keys
        ///     For a full description of these options, see set_printoptions.
        /// </param>
        /// <returns>
        ///     KeywordArgs of current print options with keys
        ///     For a full description of these options, see set_printoptions.
        /// </returns>
        public static Hashtable get_printoptions(Hashtable print_opts)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return cp.get_printoptions(print_opts);
            }
            else
            {
                return np.get_printoptions(print_opts);
            }
        }

        /*
        /// <summary>
        ///	SetKeywordArg a Python function to be used when pretty printing arrays.
        /// </summary>
        /// <param name="f">
        ///	Function to be used to pretty print arrays.<br></br>
        ///	The function should expect
        ///	a single array argument and return a string of the representation of
        ///	the array.<br></br>
        ///	If None, the function is reset to the default Cupy function
        ///	to print arrays.
        /// </param>
        /// <param name="repr">
        ///	If True (default), the function for pretty printing (__repr__)
        ///	is set, if False the function that returns the default string
        ///	representation (__str__) is set.
        /// </param>
        public static void set_string_function(function or None f, bool? repr = true)
        {
            //auto-generated code, do not change
            var __self__=self;
            var pyargs=ToTuple(new object[]
            {
                f,
            });
            var kwargs=new PyDict();
            if (repr!=true) kwargs["repr"]=ToPython(repr);
            dynamic py = __self__.InvokeMethod("set_string_function", pyargs, kwargs);
        }
        */

        /// <summary>
        ///     Return a string representation of a number in the given base system.
        /// </summary>
        /// <param name="number">
        ///     The value to convert.<br></br>
        ///     Positive and negative values are handled.
        /// </param>
        /// <param name="base">
        ///     Convert number to the base number system.<br></br>
        ///     The valid range is 2-36,
        ///     the default value is 2.
        /// </param>
        /// <param name="padding">
        ///     Number of zeros padded on the left.<br></br>
        ///     Default is 0 (no padding).
        /// </param>
        /// <returns>
        ///     String representation of number in base system.
        /// </returns>
        public static string base_repr(int number, int? @base = 2, int? padding = 0)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return cp.base_repr(number, @base, padding);
            }
            else
            {
                return np.base_repr(number, @base, padding);
            }
        }

        /// <summary>
        ///     A generic data source file (file, http, ftp, …).<br></br>
        ///     DataSources can be local files or remote files/URLs.<br></br>
        ///     The files may
        ///     also be compressed or uncompressed.<br></br>
        ///     DataSource hides some of the
        ///     low-level details of downloading the file, allowing you to simply pass
        ///     in a valid file path (or URL) and obtain a file object.<br></br>
        ///     Notes
        ///     URLs require a scheme string (http://) to be used, without it they
        ///     will fail:
        ///     Temporary directories are deleted when the DataSource is deleted.
        /// </summary>
        /// <param name="destpath">
        ///     Path to the directory where the source file gets downloaded to for
        ///     use.<br></br>
        ///     If destpath is None, a temporary directory will be created.<br></br>
        ///     The default path is the current directory.
        /// </param>
        public static void DataSource(string destpath = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                cp.DataSource(destpath);
            }
            else
            {
                np.DataSource(destpath);
            }
        }
    }
}
