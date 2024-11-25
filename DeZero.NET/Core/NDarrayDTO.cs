using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json.Serialization;
using System.Threading.Tasks;
using Cupy;

namespace DeZero.NET.Core
{
    public class NDarrayDTO
    {
        [JsonPropertyName("data")]
        public object[] Data { get; set; }

        [JsonPropertyName("dtype")]
        public string dtype { get; set; }

        [JsonPropertyName("ndim")]
        public int ndim { get; set; }

        [JsonPropertyName("shape")]
        public int[] shape { get; set; }

        public static NDarrayDTO FromNDarray(NDarray ndarray)
        {
            using var ndarray_dtype = ndarray.dtype;
            using var _float16 = Dtype.float16;
            using var _float32 = Dtype.float32;
            using var _float64 = Dtype.float64;
            using var _int8 = Dtype.int8;
            using var _int16 = Dtype.int16;
            using var _int32 = Dtype.int32;
            using var _int64 = Dtype.int64;
            using var _uint8 = Dtype.uint8;
            using var _uint16 = Dtype.uint16;
            using var _uint32 = Dtype.uint32;
            using var _uint64 = Dtype.uint64;
            if (ndarray_dtype == _float16 || ndarray_dtype == _float32)
            {
                if (ndarray.ndim == 1)
                {
                    return new NDarrayDTO
                    {
                        Data = ndarray.GetData<float[]>().Cast<object>().ToArray(),
                        dtype = ndarray_dtype.ToString(),
                        ndim = ndarray.ndim
                    };
                }
                else if (ndarray.ndim == 2)
                {
                    return new NDarrayDTO
                    {
                        Data = ndarray.GetData<float[][]>().SelectMany(x => x).Cast<object>().ToArray(),
                        dtype = ndarray_dtype.ToString(),
                        ndim = ndarray.ndim
                    };
                }
                else if (ndarray.ndim == 3)
                {
                    return new NDarrayDTO
                    {
                        Data = ndarray.GetData<float[][][]>().SelectMany(x => x).SelectMany(x => x).Cast<object>().ToArray(),
                        dtype = ndarray_dtype.ToString(),
                        ndim = ndarray.ndim
                    };
                }
                else if (ndarray.ndim == 4)
                {
                    return new NDarrayDTO
                    {
                        Data = ndarray.GetData<float[][][][]>().SelectMany(x => x).SelectMany(x => x).SelectMany(x => x).Cast<object>().ToArray(),
                        dtype = ndarray_dtype.ToString(),
                        ndim = ndarray.ndim
                    };
                }
                else
                {
                    throw new Exception("Unsupported ndim");
                }
            }
            else if (ndarray_dtype == _float64)
            {
                if (ndarray.ndim == 1)
                {
                    return new NDarrayDTO
                    {
                        Data = ndarray.GetData<double[]>().Cast<object>().ToArray(),
                        dtype = ndarray_dtype.ToString(),
                        ndim = ndarray.ndim
                    };
                }
                else if (ndarray.ndim == 2)
                {
                    return new NDarrayDTO
                    {
                        Data = ndarray.GetData<double[][]>().SelectMany(x => x).Cast<object>().ToArray(),
                        dtype = ndarray_dtype.ToString(),
                        ndim = ndarray.ndim
                    };
                }
                else if (ndarray.ndim == 3)
                {
                    return new NDarrayDTO
                    {
                        Data = ndarray.GetData<double[][][]>().SelectMany(x => x).SelectMany(x => x).Cast<object>().ToArray(),
                        dtype = ndarray_dtype.ToString(),
                        ndim = ndarray.ndim
                    };
                }
                else if (ndarray.ndim == 4)
                {
                    return new NDarrayDTO
                    {
                        Data = ndarray.GetData<double[][][][]>().SelectMany(x => x).SelectMany(x => x).SelectMany(x => x).Cast<object>().ToArray(),
                        dtype = ndarray_dtype.ToString(),
                        ndim = ndarray.ndim
                    };
                }
                else
                {
                    throw new Exception("Unsupported ndim");
                }
            }
            else if (ndarray_dtype == _int8 || ndarray_dtype == _int16 || ndarray_dtype == _int32 ||
                     ndarray_dtype == _int64)
            {
                if (ndarray.ndim == 1)
                {
                    return new NDarrayDTO
                    {
                        Data = ndarray.GetData<int[]>().Cast<object>().ToArray(),
                        dtype = ndarray_dtype.ToString(),
                        ndim = ndarray.ndim
                    };
                }
                else if (ndarray.ndim == 2)
                {
                    return new NDarrayDTO
                    {
                        Data = ndarray.GetData<int[][]>().SelectMany(x => x).Cast<object>().ToArray(),
                        dtype = ndarray_dtype.ToString(),
                        ndim = ndarray.ndim
                    };
                }
                else if (ndarray.ndim == 3)
                {
                    return new NDarrayDTO
                    {
                        Data = ndarray.GetData<int[][][]>().SelectMany(x => x).SelectMany(x => x).Cast<object>().ToArray(),
                        dtype = ndarray_dtype.ToString(),
                        ndim = ndarray.ndim
                    };
                }
                else if (ndarray.ndim == 4)
                {
                    return new NDarrayDTO
                    {
                        Data = ndarray.GetData<int[][][][]>().SelectMany(x => x).SelectMany(x => x).SelectMany(x => x).Cast<object>().ToArray(),
                        dtype = ndarray_dtype.ToString(),
                        ndim = ndarray.ndim
                    };
                }
                else
                {
                    throw new Exception("Unsupported ndim");
                }
            }
            else if (ndarray_dtype == _uint8 || ndarray_dtype == _uint16 || ndarray_dtype == _uint32 ||
                     ndarray_dtype == _uint64)
            {
                if (ndarray.ndim == 1)
                {
                    return new NDarrayDTO
                    {
                        Data = ndarray.GetData<uint[]>().Cast<object>().ToArray(),
                        dtype = ndarray_dtype.ToString(),
                        ndim = ndarray.ndim
                    };
                }
                else if (ndarray.ndim == 2)
                {
                    return new NDarrayDTO
                    {
                        Data = ndarray.GetData<uint[][]>().SelectMany(x => x).Cast<object>().ToArray(),
                        dtype = ndarray_dtype.ToString(),
                        ndim = ndarray.ndim
                    };
                }
                else if (ndarray.ndim == 3)
                {
                    return new NDarrayDTO
                    {
                        Data = ndarray.GetData<uint[][][]>().SelectMany(x => x).SelectMany(x => x).Cast<object>().ToArray(),
                        dtype = ndarray_dtype.ToString(),
                        ndim = ndarray.ndim
                    };
                }
                else if (ndarray.ndim == 4)
                {
                    return new NDarrayDTO
                    {
                        Data = ndarray.GetData<uint[][][][]>().SelectMany(x => x).SelectMany(x => x).SelectMany(x => x).Cast<object>().ToArray(),
                        dtype = ndarray_dtype.ToString(),
                        ndim = ndarray.ndim
                    };
                }
                else
                {
                    throw new Exception("Unsupported ndim");
                }
            }
            else
            {
                throw new Exception("Unsupported dtype");
            }
        }

        public NDarray ToNDarray()
        {
            if (dtype == "float16" || dtype == "float32")
            {
                return xp.array(Data.Cast<float>().ToArray()).reshape(shape);
            }
            else if (dtype == "float64")
            {
                return xp.array(Data.Cast<double>().ToArray()).reshape(shape);
            }
            else if (dtype == "int8" || dtype == "int16" || dtype == "int32" ||
                     dtype == "int64")
            {
                return xp.array(Data.Cast<int>().ToArray()).reshape(shape);
            }
            else if (dtype == "uint8" || dtype == "uint16" || dtype == "uint32" ||
                     dtype == "uint64")
            {
                return xp.array(Data.Cast<uint>().ToArray()).reshape(shape);
            }
            else
            {
                throw new Exception("Unsupported dtype");
            }
        }
        
    }

    public class NDarrayDTO<T> where T : struct
    {
        [JsonPropertyName("data")]
        public T[] Data { get; set; }

        public static NDarrayDTO<T> FromNDarray(NDarray<T> ndarray)
        {
            return new NDarrayDTO<T>
            {
                Data = ndarray.GetData<T[]>()
            };
        }

        public NDarray<T> ToNDarray()
        {
            return new NDarray<T>(Data);
        }
    }
}
