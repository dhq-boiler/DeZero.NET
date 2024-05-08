using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DeZero.NET.Core;
using DeZero.NET.Functions;
using DeZero.NET.Tests.Chainer;

namespace DeZero.NET.Tests
{
    public class Deconv2dTests
    {
        public class cp
        {
            [OneTimeSetUp]
            public void OneTimeSetUp()
            {
                if (string.IsNullOrEmpty(Runtime.PythonDLL))
                {
                    Runtime.PythonDLL = @"C:\Users\boiler\AppData\Local\Programs\Python\Python38\python38.dll";
                    PythonEngine.Initialize();
                }
            }

            [SetUp]
            public void Setup()
            {
                Gpu.Use = true;
            }

            [Test]
            public void Test_Forward1()
            {
                int n = 10, c_i = 1, c_o = 3;
                int h_i = 5, w_i = 10;
                int h_k = 10, w_k = 10;
                int h_p = 5, w_p = 5;
                int s_y = 5, s_x = 5;
                var x = xp.random.uniform(new NDarray<float>(0), new NDarray<float>(1), [n, c_i, h_i, w_i]).astype(xp.float32);
                var W = xp.random.uniform(new NDarray<float>(0), new NDarray<float>(1), [c_i, c_o, h_k, w_k]).astype(xp.float32);
                var b = xp.random.uniform(new NDarray<float>(0), new NDarray<float>(1), [c_o]).astype(xp.float32);
                var expected = Chainer.CF.deconvolution_2d(x, W, b, stride: (s_y, s_x), pad: (h_p, w_p));
                var y = Deconv2d.Invoke(x.ToVariable(), W.ToVariable(), b?.ToVariable(), stride: (s_y, s_x), pad: (h_p, w_p));
                Assert.IsTrue(Utils.array_allclose(expected, y[0].Data));
            }

            [Test]
            public void Test_Forward2()
            {
                int n = 10, c_i = 1, c_o = 3;
                int h_i = 5, w_i = 10;
                int h_k = 10, w_k = 10;
                int h_p = 5, w_p = 5;
                int s_y = 5, s_x = 5;
                var x = xp.random.uniform(new NDarray<float>(0), new NDarray<float>(1), [n, c_i, h_i, w_i]).astype(xp.float32);
                var W = xp.random.uniform(new NDarray<float>(0), new NDarray<float>(1), [c_i, c_o, h_k, w_k]).astype(xp.float32);
                NDarray b = null;
                var expected = Chainer.CF.deconvolution_2d(x, W, b, stride: (s_y, s_x), pad: (h_p, w_p));
                var y = Deconv2d.Invoke(x.ToVariable(), W.ToVariable(), b?.ToVariable(), stride: (s_y, s_x), pad: (h_p, w_p));
                Assert.IsTrue(Utils.array_allclose(expected, y[0].Data));
            }

            [Test]
            public void Test_Backward1()
            {
                int n = 10, c_i = 1, c_o = 3;
                int h_i = 5, w_i = 10;
                int h_k = 10, w_k = 10;
                int h_p = 5, w_p = 5;
                int s_y = 5, s_x = 5;
                var x = xp.random.uniform(new NDarray<float>(0), new NDarray<float>(1), [n, c_i, h_i, w_i]);
                var W = xp.random.uniform(new NDarray<float>(0), new NDarray<float>(1), [c_i, c_o, h_k, w_k]);
                NDarray b = null;
                Func<Params, Variable[]> f = x => Deconv2d.Invoke(x.Get<Variable>("x"), W.ToVariable(), b?.ToVariable(), stride: (s_y, s_x), pad: (h_p, w_p));
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetKeywordArg(x)));
            }
        }
    }
}
