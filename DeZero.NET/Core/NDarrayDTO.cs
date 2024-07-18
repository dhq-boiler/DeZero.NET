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
            if (ndarray.dtype == Dtype.float16 || ndarray.dtype == Dtype.float32)
            {
                if (ndarray.ndim == 1)
                {
                    return new NDarrayDTO
                    {
                        Data = ndarray.GetData<float[]>().Cast<object>().ToArray(),
                        dtype = ndarray.dtype.ToString(),
                        ndim = ndarray.ndim
                    };
                }
                else if (ndarray.ndim == 2)
                {
                    return new NDarrayDTO
                    {
                        Data = ndarray.GetData<float[][]>().SelectMany(x => x).Cast<object>().ToArray(),
                        dtype = ndarray.dtype.ToString(),
                        ndim = ndarray.ndim
                    };
                }
                else if (ndarray.ndim == 3)
                {
                    return new NDarrayDTO
                    {
                        Data = ndarray.GetData<float[][][]>().SelectMany(x => x).SelectMany(x => x).Cast<object>().ToArray(),
                        dtype = ndarray.dtype.ToString(),
                        ndim = ndarray.ndim
                    };
                }
                else if (ndarray.ndim == 4)
                {
                    return new NDarrayDTO
                    {
                        Data = ndarray.GetData<float[][][][]>().SelectMany(x => x).SelectMany(x => x).SelectMany(x => x).Cast<object>().ToArray(),
                        dtype = ndarray.dtype.ToString(),
                        ndim = ndarray.ndim
                    };
                }
                else
                {
                    throw new Exception("Unsupported ndim");
                }
            }
            else if (ndarray.dtype == Dtype.float64)
            {
                if (ndarray.ndim == 1)
                {
                    return new NDarrayDTO
                    {
                        Data = ndarray.GetData<double[]>().Cast<object>().ToArray(),
                        dtype = ndarray.dtype.ToString(),
                        ndim = ndarray.ndim
                    };
                }
                else if (ndarray.ndim == 2)
                {
                    return new NDarrayDTO
                    {
                        Data = ndarray.GetData<double[][]>().SelectMany(x => x).Cast<object>().ToArray(),
                        dtype = ndarray.dtype.ToString(),
                        ndim = ndarray.ndim
                    };
                }
                else if (ndarray.ndim == 3)
                {
                    return new NDarrayDTO
                    {
                        Data = ndarray.GetData<double[][][]>().SelectMany(x => x).SelectMany(x => x).Cast<object>().ToArray(),
                        dtype = ndarray.dtype.ToString(),
                        ndim = ndarray.ndim
                    };
                }
                else if (ndarray.ndim == 4)
                {
                    return new NDarrayDTO
                    {
                        Data = ndarray.GetData<double[][][][]>().SelectMany(x => x).SelectMany(x => x).SelectMany(x => x).Cast<object>().ToArray(),
                        dtype = ndarray.dtype.ToString(),
                        ndim = ndarray.ndim
                    };
                }
                else
                {
                    throw new Exception("Unsupported ndim");
                }
            }
            else if (ndarray.dtype == Dtype.int8 || ndarray.dtype == Dtype.int16 || ndarray.dtype == Dtype.int32 ||
                     ndarray.dtype == Dtype.int64)
            {
                if (ndarray.ndim == 1)
                {
                    return new NDarrayDTO
                    {
                        Data = ndarray.GetData<int[]>().Cast<object>().ToArray(),
                        dtype = ndarray.dtype.ToString(),
                        ndim = ndarray.ndim
                    };
                }
                else if (ndarray.ndim == 2)
                {
                    return new NDarrayDTO
                    {
                        Data = ndarray.GetData<int[][]>().SelectMany(x => x).Cast<object>().ToArray(),
                        dtype = ndarray.dtype.ToString(),
                        ndim = ndarray.ndim
                    };
                }
                else if (ndarray.ndim == 3)
                {
                    return new NDarrayDTO
                    {
                        Data = ndarray.GetData<int[][][]>().SelectMany(x => x).SelectMany(x => x).Cast<object>().ToArray(),
                        dtype = ndarray.dtype.ToString(),
                        ndim = ndarray.ndim
                    };
                }
                else if (ndarray.ndim == 4)
                {
                    return new NDarrayDTO
                    {
                        Data = ndarray.GetData<int[][][][]>().SelectMany(x => x).SelectMany(x => x).SelectMany(x => x).Cast<object>().ToArray(),
                        dtype = ndarray.dtype.ToString(),
                        ndim = ndarray.ndim
                    };
                }
                else
                {
                    throw new Exception("Unsupported ndim");
                }
            }
            else if (ndarray.dtype == Dtype.uint8 || ndarray.dtype == Dtype.uint16 || ndarray.dtype == Dtype.uint32 ||
                     ndarray.dtype == Dtype.uint64)
            {
                if (ndarray.ndim == 1)
                {
                    return new NDarrayDTO
                    {
                        Data = ndarray.GetData<uint[]>().Cast<object>().ToArray(),
                        dtype = ndarray.dtype.ToString(),
                        ndim = ndarray.ndim
                    };
                }
                else if (ndarray.ndim == 2)
                {
                    return new NDarrayDTO
                    {
                        Data = ndarray.GetData<uint[][]>().SelectMany(x => x).Cast<object>().ToArray(),
                        dtype = ndarray.dtype.ToString(),
                        ndim = ndarray.ndim
                    };
                }
                else if (ndarray.ndim == 3)
                {
                    return new NDarrayDTO
                    {
                        Data = ndarray.GetData<uint[][][]>().SelectMany(x => x).SelectMany(x => x).Cast<object>().ToArray(),
                        dtype = ndarray.dtype.ToString(),
                        ndim = ndarray.ndim
                    };
                }
                else if (ndarray.ndim == 4)
                {
                    return new NDarrayDTO
                    {
                        Data = ndarray.GetData<uint[][][][]>().SelectMany(x => x).SelectMany(x => x).SelectMany(x => x).Cast<object>().ToArray(),
                        dtype = ndarray.dtype.ToString(),
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
