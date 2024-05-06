using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeZero.NET.Tests
{
    public class TestConv2d_simple
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
                int n = 1, c = 5, h = 15, w = 15;
                int o = 8;
                (int, int) k = (3, 3), s = (1, 1), p = (1, 1);
                var x = xp.random.randn(n, c, h, w).astype("f");
                var W = xp.random.randn(o, c, k.Item1, k.Item2).astype("f");
                NDarray b = null;
                var y = Utils.conv2d_simple(x, W, b, s, p);
                var expected = Chainer.CF.convolution_2d(x, W, b, s, p);
                Assert.IsTrue(Utils.array_allclose(expected, y));
            }
        }

        public class np
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
                Gpu.Use = false;
            }

            [Test]
            public void Test_Forward1()
            {
                int n = 1, c = 5, h = 15, w = 15;
                int o = 8;
                (int, int) k = (3, 3), s = (1, 1), p = (1, 1);
                var x = xp.random.randn(n, c, h, w).astype("f");
                var W = xp.random.randn(o, c, k.Item1, k.Item2).astype("f");
                NDarray b = null;
                var y = Utils.conv2d_simple(x, W, b, s, p);
                var expected = Chainer.CF.convolution_2d(x, W, b, s, p);
                Assert.IsTrue(Utils.array_allclose(expected, y));
            }
        }
    }
}
