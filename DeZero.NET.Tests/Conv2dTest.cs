﻿using DeZero.NET.Core;
using Python.Runtime;

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
                var y = Utils.conv2d_simple(x.ToVariable(), W, b, s, p);
                var expected = Chainer.CF.convolution_2d(x, W, b, s, p);
                Assert.IsTrue(Utils.array_allclose(expected, y.Data));
            }

            [Test]
            public void Test_Forward2()
            {
                int n = 1, c = 5, h = 15, w = 15;
                int o = 8;
                (int, int) k = (3, 3), s = (3, 1), p = (2, 1);
                var x = xp.random.randn(n, c, h, w).astype("f");
                var W = xp.random.randn(o, c, k.Item1, k.Item2).astype("f"); 
                NDarray b = null;
                var y = Utils.conv2d_simple(x.ToVariable(), W, b, s, p);
                var expected = Chainer.CF.convolution_2d(x, W, b, s, p);
                Assert.IsTrue(Utils.array_allclose(expected, y.Data));
            }

            [Test]
            public void Test_Forward3()
            {
                int n = 1, c = 5, h = 20, w = 15;
                int o = 3;
                (int, int) k = (5, 3);
                int s = 1, p = 3;
                var x = xp.random.randn(n, c, h, w).astype("f");
                var W = xp.random.randn(o, c, k.Item1, k.Item2).astype("f");
                NDarray b = null;
                var y = Utils.conv2d_simple(x.ToVariable(), W, b, s, p);
                var expected = Chainer.CF.convolution_2d(x, W, b, s, p);
                Assert.IsTrue(Utils.array_allclose(expected, y.Data));
            }

            [Test]
            public void Test_Forward4()
            {
                int n = 1, c = 5, h = 20, w = 15;
                int o = 3;
                (int, int) k = (5, 3);
                int s = 1, p = 3;
                var x = xp.random.randn(n, c, h, w).astype("f");
                var W = xp.random.randn(o, c, k.Item1, k.Item2).astype("f");
                var b = xp.random.randn(o).astype("f");
                var y = Utils.conv2d_simple(x.ToVariable(), W, b, s, p);
                var expected = Chainer.CF.convolution_2d(x, W, b, s, p);
                Assert.IsTrue(Utils.array_allclose(expected, y.Data));
            }

            [Test]
            public void Test_Backward1()
            {
                int n = 1, c = 5, h = 20, w = 15;
                int o = 3;
                (int, int) k = (5, 3);
                int s = 1, p = 3;
                var x = xp.random.randn(n, c, h, w).astype("f");
                var W = xp.random.randn(o, c, k.Item1, k.Item2).astype("f");
                var b = xp.random.randn(o);
                var f = new Func<Params, Variable[]>(args => [Utils.conv2d_simple(args.Get<Variable>("x"), W, b, s, p)]);
                Assert.That(Utils.gradient_check(new Function(f), Params<Variable>.args(x.ToVariable(), "x"), Params.Empty));
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
                var y = Utils.conv2d_simple(x.ToVariable(), W, b, s, p);
                var expected = Chainer.CF.convolution_2d(x, W, b, s, p);
                Assert.IsTrue(Utils.array_allclose(expected, y.Data));
            }

            [Test]
            public void Test_Forward2()
            {
                int n = 1, c = 5, h = 15, w = 15;
                int o = 8;
                (int, int) k = (3, 3), s = (3, 1), p = (2, 1);
                var x = xp.random.randn(n, c, h, w).astype("f");
                var W = xp.random.randn(o, c, k.Item1, k.Item2).astype("f");
                NDarray b = null;
                var y = Utils.conv2d_simple(x.ToVariable(), W, b, s, p);
                var expected = Chainer.CF.convolution_2d(x, W, b, s, p);
                Assert.IsTrue(Utils.array_allclose(expected, y.Data));
            }

            [Test]
            public void Test_Forward3()
            {
                int n = 1, c = 5, h = 20, w = 15;
                int o = 3;
                (int, int) k = (5, 3);
                int s = 1, p = 3;
                var x = xp.random.randn(n, c, h, w).astype("f");
                var W = xp.random.randn(o, c, k.Item1, k.Item2).astype("f");
                NDarray b = null;
                var y = Utils.conv2d_simple(x.ToVariable(), W, b, s, p);
                var expected = Chainer.CF.convolution_2d(x, W, b, s, p);
                Assert.IsTrue(Utils.array_allclose(expected, y.Data));
            }

            [Test]
            public void Test_Backward1()
            {
                int n = 1, c = 5, h = 20, w = 15;
                int o = 3;
                (int, int) k = (5, 3);
                int s = 1, p = 3;
                var x = xp.random.randn(n, c, h, w).astype("f");
                var W = xp.random.randn(o, c, k.Item1, k.Item2).astype("f");
                var b = xp.random.randn(o);
                var f = new Func<Params, Variable[]>(args => [Utils.conv2d_simple(args.Get<Variable>("x"), W, b, s, p)]);
                Assert.That(Utils.gradient_check(new Function(f), Params<Variable>.args(x.ToVariable(), "x"), Params.Empty));
            }
        }
    }
}