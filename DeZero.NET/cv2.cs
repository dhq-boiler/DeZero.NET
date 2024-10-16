using Cupy;
using DeZero.NET.PIL;
using Microsoft.VisualBasic;
using Numpy;
using Python.Included;
using Python.Runtime;
using System.Diagnostics;
using DeZero.NET;
using DeZero.NET.Core;
using DeZero.NET.Models;
using DeZero.NET.OpenCv;
using OpenCvSharp;

namespace DeZero.NET
{
    namespace OpenCv
    {
        public static partial class Cv2
        {
            private static Lazy<PyObject> _lazy_self;

            static Cv2()
            {
                ReInitializeLazySelf();
            }

            public static PyObject self => _lazy_self.Value;

            public static dynamic dynamic_self => self;
            private static bool IsInitialized => self != null;

            private static void ReInitializeLazySelf()
            {
                _lazy_self = new Lazy<PyObject>(() =>
                    {
                        try
                        {
                            return InstallAndImport();
                        }
                        catch (Exception)
                        {
                            // retry to fix the installation by forcing a repair, if Python.Included is used.
                            return InstallAndImport(true);
                        }
                    }
                );
            }

            private static PyObject InstallAndImport(bool force = false)
            {
#if PYTHON_INCLUDED
            Installer.SetupPython(force).Wait();
#endif
#if !PYTHON_INCLUDED
                PythonEngine.Initialize();

                //if (!Installer.IsModuleInstalled("numpy"))
                //{
                //    Installer.PipInstallModule("numpy", "1.24.2", true).GetAwaiter().GetResult();
                //}

                //if (!Installer.IsModuleInstalled("cupy"))
                //{
                //    Installer.PipInstallModule("cupy_cuda12x", force: true).GetAwaiter().GetResult();
                //}

                if (!Installer.IsModuleInstalled("opencv-python"))
                {
                    Installer.PipInstallModule("opencv-python", force: true).GetAwaiter().GetResult();
                }

#endif
                PythonEngine.AddShutdownHandler(() => ReInitializeLazySelf());
                PythonEngine.Initialize();
                var cv2 = Py.Import("cv2");
                return cv2;
            }


            public static void Dispose()
            {
                self?.Dispose();
            }


            //auto-generated
            internal static PyTuple ToTuple(Array input)
            {
                var array = new PyObject[input.Length];
                for (var i = 0; i < input.Length; i++) array[i] = ToPython(input.GetValue(i));
                return new PyTuple(array);
            }

            //auto-generated
            internal static PyObject ToPython(object obj)
            {
                if (obj == null) return Runtime.None;
                switch (obj)
                {
                    // basic types
                    case int o: return new PyInt(o);
                    case long o: return new PyInt(o);
                    case float o: return new PyFloat(o);
                    case double o: return new PyFloat(o);
                    case string o: return new PyString(o);
                    case bool o: return o.ToPython();
                    case ValueTuple<int, int> o:
                        return new PyTuple([new PyInt((((int, int))o).Item1), new PyInt((((int, int))o).Item2)]);
                    case ValueTuple<int, int, int> o:
                        return new PyTuple([new PyInt((((int, int, int))o).Item1), new PyInt((((int, int, int))o).Item2), new PyInt((((int, int, int))o).Item3)]);
                    case ValueTuple<int, int, int, int> o:
                        return new PyTuple([new PyInt((((int, int, int, int))o).Item1), new PyInt((((int, int, int, int))o).Item2), new PyInt((((int, int, int, int))o).Item3), new PyInt((((int, int, int, int))o).Item4)]);
                    case PyObject o: return o;
                    // sequence types
                    case Array o: return ToTuple(o);
                    // special types from 'ToPythonConversions'
                    case Cupy.Models.Axis o: return o.Axes == null ? null : ToTuple(o.Axes);
                    case Numpy.Models.Axis o: return o.Axes == null ? null : ToTuple(o.Axes);
                    case Cupy.Models.Shape o: return ToTuple(o.Dimensions);
                    case Numpy.Models.Shape o: return ToTuple(o.Dimensions);
                    case Cupy.Models.Slice o: return o.ToPython();
                    case Numpy.Models.Slice o: return o.ToPython();
                    case Cupy.PythonObject o: return o.PyObject;
                    case Numpy.PythonObject o: return o.PyObject;
                    case Dictionary<string, NDarray> o: return ToDict(o);
                    default:
                        throw new NotImplementedException(
                            $"Type is not yet supported: {obj.GetType().Name}. Add it to 'ToPythonConversions'");
                }
            }

            //auto-generated
            internal static T ToCsharp<T>(dynamic pyobj)
            {
                switch (typeof(T).Name)
                {
                    // types from 'ToCsharpConversions'
                    case "Mat": return (T)(object)new Mat(pyobj);
                    case "Dtype": return (T)(object)new Dtype(pyobj);
                    case "NDarray": return (T)(object)new NDarray(pyobj);
                    case "NDarray`1":
                        switch (typeof(T).GenericTypeArguments[0].Name)
                        {
                            case "Byte": return (T)(object)new NDarray<byte>(pyobj);
                            case "Short": return (T)(object)new NDarray<short>(pyobj);
                            case "Boolean": return (T)(object)new NDarray<bool>(pyobj);
                            case "Int32": return (T)(object)new NDarray<int>(pyobj);
                            case "Int64": return (T)(object)new NDarray<long>(pyobj);
                            case "Single": return (T)(object)new NDarray<float>(pyobj);
                            case "Double": return (T)(object)new NDarray<double>(pyobj);
                            default:
                                throw new NotImplementedException(
                                    $"Type NDarray<{typeof(T).GenericTypeArguments[0].Name}> missing. Add it to 'ToCsharpConversions'");
                        }

                        break;
                    case "NDarray[]":
                        var po = pyobj as PyObject;
                        var len = po.Length();
                        var rv = new NDarray[len];
                        for (var i = 0; i < len; i++)
                            rv[i] = ToCsharp<NDarray>(po[i]);
                        return (T)(object)rv;
                    case "Matrix": return (T)(object)new Matrix(pyobj);
                    default:
                        var pyClass = $"{pyobj.__class__}";
                        if (pyClass == "<class 'str'>") return (T)(object)pyobj.ToString();
                        if (pyClass.StartsWith("<class 'Pillow")) return (pyobj.item() as PyObject).As<T>();
                        if (pyClass.StartsWith("<class 'PIL")) return (pyobj.item() as PyObject).As<T>();
                        try
                        {
                            return pyobj.As<T>();
                        }
                        catch (Exception e)
                        {
                            throw new NotImplementedException(
                                $"conversion from {pyobj.__class__} to {typeof(T).Name} not implemented", e);
                            return default;
                        }
                }
            }

            //auto-generated
            internal static T SharpToSharp<T>(object obj)
            {
                if (obj == null) return default;
                switch (obj)
                {
                    // from 'SharpToSharpConversions':
                    case Array a:
                        if (typeof(T) == typeof(NDarray)) return (T)(object)ConvertArrayToNDarray(a);
                        break;
                }

                throw new NotImplementedException(
                    $"Type is not yet supported: {obj.GetType().Name}. Add it to 'SharpToSharpConversions'");
            }

            //auto-generated: SpecialConversions
            private static NDarray ConvertArrayToNDarray(Array a)
            {
                switch (a)
                {
                    case bool[] arr: return xp.array(arr);
                    case int[] arr: return xp.array(arr);
                    case float[] arr: return xp.array(arr);
                    case double[] arr: return xp.array(arr);
                    case int[,] arr:
                        return xp.array(arr.Cast<int>().ToArray()).reshape(arr.GetLength(0), arr.GetLength(1));
                    case float[,] arr:
                        return xp.array(arr.Cast<float>().ToArray()).reshape(arr.GetLength(0), arr.GetLength(1));
                    case double[,] arr:
                        return xp.array(arr.Cast<double>().ToArray()).reshape(arr.GetLength(0), arr.GetLength(1));
                    case bool[,] arr:
                        return xp.array(arr.Cast<bool>().ToArray()).reshape(arr.GetLength(0), arr.GetLength(1));
                    default:
                        throw new NotImplementedException(
                            $"Type {a.GetType()} not supported yet in ConvertArrayToNDarray.");
                }
            }

            //auto-generated: SpecialConversions
            private static PyDict ToDict(Dictionary<string, NDarray> d)
            {
                var dict = new PyDict();
                foreach (var pair in d)
                {
                    using var key = new PyString(pair.Key);
                    dict[key] = pair.Value.self;
                }

                return dict;
            }

            #region Functions

            public static Image open(string fp, string mode = "r", string formats = "None")
            {
                var __self__ = self;
                using var args = ToTuple(new object[]
                {
                    fp, mode
                });
                using var py = self.InvokeMethod("open", args);
                args.Dispose();
                return ToCsharp<Image>(py);
            }

            public static NDarray resize(NDarray img, (int, int) dsize, double fx = 0, double fy = 0, int interpolation = 1)
            {
                var __self__ = self;
                using var args = ToTuple(new object[]
                {
                    img.ToNumpyNDarray.PyObject, dsize, fx, fy, interpolation
                });
                var py = self.InvokeMethod("resize", args);
                args.Dispose();
                return ToCsharp<NDarray>(py);
            }

            public static NDarray resize(NDarray img, (int, int, int) dsize, double fx = 0, double fy = 0, int interpolation = 1)
            {
                var __self__ = self;
                using var args = ToTuple(new object[]
                {
                    img.ToNumpyNDarray.PyObject, dsize, fx, fy, interpolation
                });
                var py = self.InvokeMethod("resize", args);
                args.Dispose();
                return ToCsharp<NDarray>(py);
            }

            public static NDarray resize(NDarray img, (int, int, int, int) dsize, double fx = 0, double fy = 0, int interpolation = 1)
            {
                var __self__ = self;
                using var args = ToTuple(new object[]
                {
                    img.ToNumpyNDarray.PyObject, dsize, fx, fy, interpolation
                });
                var py = self.InvokeMethod("resize", args);
                args.Dispose();
                return ToCsharp<NDarray>(py);
            }

            public const int INTER_NEAREST = 0;
            public const int INTER_LINEAR = 1;
            public const int INTER_CUBIC = 2;
            public const int INTER_AREA = 3;
            public const int INTER_LANCZOS4 = 4;

            #endregion
        }

        public partial class PythonObject : IDisposable
        {
            public PyObject self; // can not be made readonly because of NDarray(IntPtr ... )

            public PythonObject(PyObject pyobject)
            {
                self = pyobject;
            }

            public PythonObject(PythonObject t)
            {
                self = t.PyObject;
            }

            protected PythonObject()
            {
            } // required for some constructors

            public dynamic PyObject => self;

            public IntPtr Handle => self.Handle;


            /// <summary>
            ///     An object to simplify the interaction of the array with the ctypes module.
            /// </summary>
            //public PyObject ctypes => self.GetAttr("ctypes"); // TODO: wrap ctypes
            public PyObject ctypes => Cupy.ctypes.self; //.GetAttr("ctypes");

            public void Dispose()
            {
                self?.Dispose();
            }

            public override bool Equals(object obj)
            {
                if (obj == null)
                    return false;
                switch (obj)
                {
                    case PythonObject other:
                        return self.Equals(other.self);
                    case PyObject other:
                        return self.Equals(other);
                }

                return base.Equals(obj);
            }

            public override int GetHashCode()
            {
                return self.GetHashCode();
            }

            public override string ToString()
            {
                return self.ToString();
            }

            public static PythonObject Create<T>(string python_class)
            {
                throw new NotImplementedException();
            }
        }

        public class VideoCapture : PythonObject
        {
            public VideoCapture(int index)
            {
                using var args = Cv2.ToTuple(new object[] { index });
                self = Cv2.self.InvokeMethod("VideoCapture", args);
            }

            public VideoCapture(string filename)
            {
                using var args = Cv2.ToTuple(new object[] { filename });
                self = Cv2.self.InvokeMethod("VideoCapture", args);
            }

            public bool IsOpened()
            {
                using var py = self.InvokeMethod("isOpened");
                return Cv2.ToCsharp<bool>(py);
            }

            public bool Read(out NDarray frame)
            {
                using var py = self.InvokeMethod("read");
                var tuple = py.As<PyTuple>();
                bool ret = tuple[0].As<bool>();
                frame = new NDarray(tuple[1]);
                return ret;
            }

            public bool Grab()
            {
                using var py = self.InvokeMethod("grab");
                return Cv2.ToCsharp<bool>(py);
            }

            public bool Retrieve(out NDarray frame, int flag = 0)
            {
                using var args = Cv2.ToTuple(new object[] { flag });
                using var py = self.InvokeMethod("retrieve", args);
                var tuple = py.As<PyTuple>();
                bool ret = tuple[0].As<bool>();
                frame = new NDarray(tuple[1]);
                return ret;
            }

            public void Release()
            {
                self.InvokeMethod("release");
            }

            public bool Set(VideoCaptureProperties propId, double value)
            {
                return Set((int)propId, value);
            }

            public bool Set(int propId, double value)
            {
                using var args = Cv2.ToTuple(new object[] { propId, value });
                using var py = self.InvokeMethod("set", args);
                return Cv2.ToCsharp<bool>(py);
            }

            public double Get(VideoCaptureProperties propId)
            {
                return Get((int)propId);
            }

            public double Get(int propId)
            {
                using var args = Cv2.ToTuple(new object[] { propId });
                using var py = self.InvokeMethod("get", args);
                return Cv2.ToCsharp<double>(py);
            }
        }

        public enum VideoCaptureProperties
        {
            #region Basic

            /// <summary>
            /// Position in milliseconds from the file beginning
            /// </summary>
            PosMsec = 0,

            /// <summary>
            /// Position in frames (only for video files)
            /// </summary>
            PosFrames = 1,

            /// <summary>
            /// Position in relative units (0 - start of the file, 1 - end of the file)
            /// </summary>
            PosAviRatio = 2,

            /// <summary>
            /// Width of frames in the video stream (only for cameras)
            /// </summary>
            FrameWidth = 3,

            /// <summary>
            /// Height of frames in the video stream (only for cameras)
            /// </summary>
            FrameHeight = 4,

            /// <summary>
            /// Frame rate (only for cameras)
            /// </summary>
            Fps = 5,

            /// <summary>
            /// 4-character code of codec (only for cameras). 
            /// </summary>
            // ReSharper disable once InconsistentNaming
            FourCC = 6,

            /// <summary>
            /// Number of frames in the video stream
            /// </summary>
            FrameCount = 7,

            /// <summary>
            /// The format of the Mat objects returned by retrieve()
            /// </summary>
            Format = 8,

            /// <summary>
            /// A backend-specific value indicating the current capture mode
            /// </summary>
            Mode = 9,

            /// <summary>
            /// Brightness of image (only for cameras) 
            /// </summary>
            Brightness = 10,

            /// <summary>
            /// contrast of image (only for cameras) 
            /// </summary>
            Contrast = 11,

            /// <summary>
            /// Saturation of image (only for cameras) 
            /// </summary>
            Saturation = 12,

            /// <summary>
            /// hue of image (only for cameras) 
            /// </summary>
            Hue = 13,

            /// <summary>
            /// Gain of the image (only for cameras)
            /// </summary>
            Gain = 14,

            /// <summary>
            /// Exposure (only for cameras)
            /// </summary>
            Exposure = 15,

            /// <summary>
            /// Boolean flags indicating whether images should be converted to RGB
            /// </summary>
            ConvertRgb = 16,

            /// <summary>
            /// 
            /// </summary>
            WhiteBalanceBlueU = 17,

            /// <summary>
            /// TOWRITE (note: only supported by DC1394 v 2.x backend currently)
            /// </summary>
            Rectification = 18,

            /// <summary>
            /// 
            /// </summary>
            Monocrome = 19,

            /// <summary>
            /// 
            /// </summary>
            Sharpness = 20,

            /// <summary>
            /// exposure control done by camera,
            /// user can adjust refernce level using this feature
            /// </summary>
            AutoExposure = 21,

            /// <summary>
            /// 
            /// </summary>
            Gamma = 22,

            /// <summary>
            /// 
            /// </summary>
            Temperature = 23,

            /// <summary>
            /// 
            /// </summary>
            Trigger = 24,

            /// <summary>
            /// 
            /// </summary>
            TriggerDelay = 25,

            /// <summary>
            /// 
            /// </summary>
            WhiteBalanceRedV = 26,

            /// <summary>
            /// 
            /// </summary>
            Zoom = 27,

            /// <summary>
            /// 
            /// </summary>
            Focus = 28,

            /// <summary>
            /// 
            /// </summary>
            Guid = 29,

            /// <summary>
            /// 
            /// </summary>
            IsoSpeed = 30,

            /// <summary>
            /// 
            /// </summary>
            BackLight = 32,

            /// <summary>
            /// 
            /// </summary>
            Pan = 33,

            /// <summary>
            /// 
            /// </summary>
            Tilt = 34,

            /// <summary>
            /// 
            /// </summary>
            Roll = 35,

            /// <summary>
            /// 
            /// </summary>
            Iris = 36,

            /// <summary>
            /// Pop up video/camera filter dialog (note: only supported by DSHOW backend currently. Property value is ignored)
            /// </summary>
            Settings = 37,

            /// <summary>
            /// 
            /// </summary>
            BufferSize = 38,

            /// <summary>
            /// 
            /// </summary>
            AutoFocus = 39,

            /// <summary>
            /// Sample aspect ratio: num/den (num)
            /// </summary>
            SARNum = 40,

            /// <summary>
            /// Sample aspect ratio: num/den (den)
            /// </summary>
            SARDen = 41,

            /// <summary>
            /// Current backend (enum VideoCaptureAPIs). Read-only property
            /// </summary>
            Backend = 42,

            /// <summary>
            /// Video input or Channel Number (only for those cameras that support)
            /// </summary>
            Channel = 43,

            /// <summary>
            /// enable/ disable auto white-balance
            /// </summary>
            AutoWB = 44,

            /// <summary>
            /// white-balance color temperature
            /// </summary>
            WBTemperature = 45,

            /// <summary>
            /// (read-only) codec's pixel format. 4-character code - see VideoWriter::fourcc . Subset of [AV_PIX_FMT_*](https://github.com/FFmpeg/FFmpeg/blob/master/libavcodec/raw.c) or -1 if unknown
            /// </summary>
            CodecPixelFormat = 46,

            /// <summary>
            /// (read-only) Video bitrate in kbits/s
            /// </summary>
            BitRate = 47,

            /// <summary>
            /// (read-only) Frame rotation defined by stream meta (applicable for FFmpeg back-end only)
            /// </summary>
            OrientationMeta = 48,

            /// <summary>
            /// if true - rotates output frames of CvCapture considering video file's metadata  (applicable for FFmpeg back-end only) (https://github.com/opencv/opencv/issues/15499)
            /// </summary>
            OrientationAuto = 49,

            /// <summary>
            /// (open-only) Hardware acceleration type (see VideoAccelerationType). 
            /// Setting supported only via params parameter in cv::VideoCapture constructor / .open() method. 
            /// Default value is backend-specific.
            /// </summary>
            HwAcceleration = 50,

            /// <summary>
            /// (open-only) Hardware device index (select GPU if multiple available). Device enumeration is acceleration type specific.
            /// </summary>
            HwDevice = 51,

            #endregion

            #region OpenNI

            // Properties of cameras available through OpenNI interfaces

            /// <summary>
            /// 
            /// </summary>
            OpenNI_OutputMode = 100,

            /// <summary>
            /// in mm
            /// </summary>
            OpenNI_FrameMaxDepth = 101,

            /// <summary>
            /// in mm
            /// </summary>
            OpenNI_Baseline = 102,

            /// <summary>
            /// in pixels
            /// </summary>
            OpenNI_FocalLength = 103,

            /// <summary>
            /// flag that synchronizes the remapping depth map to image map
            /// by changing depth generator's view point (if the flag is "on") or
            /// sets this view point to its normal one (if the flag is "off").
            /// </summary>
            OpenNI_Registration = 104,

            /// <summary>
            /// 
            /// </summary>
            OPENNI_ApproxFrameSync = 105,

            /// <summary>
            /// 
            /// </summary>
            OPENNI_MaxBufferSize = 106,

            /// <summary>
            /// 
            /// </summary>
            OPENNI_CircleBuffer = 107,

            /// <summary>
            /// 
            /// </summary>
            OPENNI_MaxTimeDuration = 108,

            /// <summary>
            /// 
            /// </summary>
            OPENNI_GeneratorPresent = 109,

            /// <summary>
            /// 
            /// </summary>
            OPENNI2_Sync = 110,

            /// <summary>
            /// 
            /// </summary>
            OPENNI2_Mirror = 111,

            /// <summary>
            /// 
            /// </summary>
            OpenNI_DepthGenerator = 1 << 31,

            /// <summary>
            /// 
            /// </summary>
            OpenNI_ImageGenerator = 1 << 30,

            /// <summary>
            /// 
            /// </summary>
            OpenNI_ImageGeneratorPresent = OpenNI_ImageGenerator + OPENNI_GeneratorPresent,


            /// <summary>
            /// 
            /// </summary>
            OpenNI_ImageGeneratorOutputMode = OpenNI_ImageGenerator + OpenNI_OutputMode,

            /// <summary>
            /// 
            /// </summary>
            OpenNI_DepthGeneratorBaseline = OpenNI_ImageGenerator + OpenNI_Baseline,

            /// <summary>
            /// 
            /// </summary>
            OpenNI_DepthGeneratorFocalLength = OpenNI_ImageGenerator + OpenNI_FocalLength,

            /// <summary>
            /// 
            /// </summary>
            OpenNI_DepthGeneratorRegistrationON = OpenNI_ImageGenerator + OpenNI_Registration,

            #endregion

            #region GStreamer

            // Properties of cameras available through GStreamer interface

            /// <summary>
            /// default is 1
            /// </summary>
            GStreamerQueueLength = 200,

            #endregion

            #region PVAPI

            /// <summary>
            /// ip for anable multicast master mode. 0 for disable multicast
            /// </summary>
            PvAPIMulticastIP = 300,

            /// <summary>
            /// Determines how a frame is initiated
            /// </summary>
            PVAPI_FrameStartTriggerMode = 301,

            /// <summary>
            /// Horizontal sub-sampling of the image
            /// </summary>
            PVAPI_DecimationHorizontal = 302,

            /// <summary>
            /// Vertical sub-sampling of the image
            /// </summary>
            PVAPI_DecimationVertical = 303,

            /// <summary>
            /// Horizontal binning factor
            /// </summary>
            PVAPI_BinningX = 304,

            /// <summary>
            /// Vertical binning factor
            /// </summary>
            PVAPI_BinningY = 305,

            /// <summary>
            /// Pixel format
            /// </summary>
            PVAPI_PixelFormat = 306,

            #endregion

            #region XI

            // Properties of cameras available through XIMEA SDK interface

            /// <summary>
            /// Change image resolution by binning or skipping. 
            /// </summary>
            XI_Downsampling = 400,

            /// <summary>
            /// Output data format.
            /// </summary>
            XI_DataFormat = 401,

            /// <summary>
            /// Horizontal offset from the origin to the area of interest (in pixels).
            /// </summary>        
            XI_OffsetX = 402,

            /// <summary>
            /// Vertical offset from the origin to the area of interest (in pixels).
            /// </summary>
            XI_OffsetY = 403,

            /// <summary>
            /// Defines source of trigger.
            /// </summary>
            XI_TrgSource = 404,

            /// <summary>
            /// Generates an internal trigger. PRM_TRG_SOURCE must be set to TRG_SOFTWARE.
            /// </summary>
            XI_TrgSoftware = 405,

            /// <summary>
            /// Selects general purpose input 
            /// </summary>
            XI_GpiSelector = 406,

            /// <summary>
            /// Set general purpose input mode
            /// </summary>
            XI_GpiMode = 407,

            /// <summary>
            /// Get general purpose level
            /// </summary>
            XI_GpiLevel = 408,

            /// <summary>
            /// Selects general purpose output 
            /// </summary>
            XI_GpoSelector = 409,

            /// <summary>
            /// Set general purpose output mode
            /// </summary>
            XI_GpoMode = 410,

            /// <summary>
            /// Selects camera signalling LED 
            /// </summary>
            XI_LedSelector = 411,

            /// <summary>
            /// Define camera signalling LED functionality
            /// </summary>
            XI_LedMode = 412,

            /// <summary>
            /// Calculates White Balance(must be called during acquisition)
            /// </summary>
            XI_ManualWB = 413,

            /// <summary>
            /// Automatic white balance
            /// </summary>
            XI_AutoWB = 414,

            /// <summary>
            /// Automatic exposure/gain
            /// </summary>
            XI_AEAG = 415,

            /// <summary>
            /// Exposure priority (0.5 - exposure 50%, gain 50%).
            /// </summary>
            XI_ExpPriority = 416,

            /// <summary>
            /// Maximum limit of exposure in AEAG procedure
            /// </summary>
            XI_AEMaxLimit = 417,

            /// <summary>
            /// Maximum limit of gain in AEAG procedure
            /// </summary>
            XI_AGMaxLimit = 418,

            /// <summary>
            /// Average intensity of output signal AEAG should achieve(in %)
            /// </summary>
            XI_AEAGLevel = 419,

            /// <summary>
            /// Image capture timeout in milliseconds
            /// </summary>
            XI_Timeout = 420,

            #endregion

            #region iOS

            /// <summary>
            /// 
            /// </summary>
            IOS_DeviceFocus = 9001,

            /// <summary>
            /// 
            /// </summary>
            IOS_DeviceExposure = 9002,

            /// <summary>
            /// 
            /// </summary>
            IOS_DeviceFlash = 9003,

            /// <summary>
            /// 
            /// </summary>
            IOS_DeviceWhiteBalance = 9004,

            /// <summary>
            /// 
            /// </summary>
            IOS_DeviceTorch = 9005,

            #endregion

            #region GIGA

            /// <summary>
            /// 
            /// </summary>
            GIGA_FrameOffsetX = 10001,

            /// <summary>
            /// 
            /// </summary>
            GIGA_FrameOffsetY = 10002,

            /// <summary>
            /// 
            /// </summary>
            GIGA_FrameWidthMax = 10003,

            /// <summary>
            /// 
            /// </summary>
            GIGA_FrameHeightMax = 10004,

            /// <summary>
            /// 
            /// </summary>
            GIGA_FrameSensWidth = 10005,

            /// <summary>
            /// 
            /// </summary>
            GIGA_FrameSensHeight = 10006,

            #endregion

            #region INTELPERC

            /// <summary>
            /// 
            /// </summary>
            INTELPERC_ProfileCount = 11001,

            /// <summary>
            /// 
            /// </summary>
            INTELPERC_ProfileIdx = 11002,

            /// <summary>
            /// 
            /// </summary>
            INTELPERC_DepthLowConfidenceValue = 11003,

            /// <summary>
            /// 
            /// </summary>
            INTELPERC_DepthSaturationValue = 11004,

            /// <summary>
            /// 
            /// </summary>
            INTELPERC_DepthConfidenceThreshold = 11005,

            /// <summary>
            /// 
            /// </summary>
            INTELPERC_DepthFocalLengthHorz = 11006,

            /// <summary>
            /// 
            /// </summary>
            INTELPERC_DepthFocalLengthVert = 11007,

            #endregion

            #region gPhoto2

            /// <summary>
            /// Capture only preview from liveview mode.
            /// </summary>
            GPhoto2_Preview = 17001,

            /// <summary>
            /// Readonly, returns (const char *).
            /// </summary>
            GPhoto2_WidgetEnumerate = 17002,

            /// <summary>
            /// Trigger, only by set. Reload camera settings.
            /// </summary>
            GPhoto2_ReloadConfig = 17003,

            /// <summary>
            /// Reload all settings on set.
            /// </summary>
            GPhoto2_ReloadOnChange = 17004,

            /// <summary>
            /// Collect messages with details.
            /// </summary>
            GPhoto2_CollectMsgs = 17005,

            /// <summary>
            /// Readonly, returns (const char *).
            /// </summary>
            GPhoto2_FlushMsgs = 17006,

            /// <summary>
            /// Exposure speed. Can be readonly, depends on camera program.
            /// </summary>
            Speed = 17007,

            /// <summary>
            /// Aperture. Can be readonly, depends on camera program.
            /// </summary>
            Aperture = 17008,

            /// <summary>
            /// Camera exposure program.
            /// </summary>
            ExposureProgram = 17009,

            /// <summary>
            /// Enter liveview mode.
            /// </summary>
            ViewFinder = 17010,

            #endregion
        }
    }
}
