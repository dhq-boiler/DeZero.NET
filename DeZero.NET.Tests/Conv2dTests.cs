using DeZero.NET.Core;
using DeZero.NET.Extensions;
using DeZero.NET.Functions;
using Python.Runtime;

namespace DeZero.NET.Tests
{
    public class Conv2d_simple_Tests
    {
        [Category("cupy")]
        public class cp
        {
            [OneTimeSetUp]
            public void OneTimeSetUp()
            {
                if (string.IsNullOrEmpty(Runtime.PythonDLL))
                {
                    Runtime.PythonDLL = @"C:\Users\boiler\AppData\Local\Programs\Python\Python311\python311.dll";
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
                var x = xp.random.randn(n, c, h, w).astype("f").ToVariable();
                var W = xp.random.randn(o, c, k.Item1, k.Item2).astype("f").ToVariable();
                Variable b = null;
                var y = Utils.conv2d_simple(x, W, b, s, p);
                var expected = Chainer.CF.convolution_2d(x.Data.Value, W.Data.Value, b?.Data.Value, s, p);
                Assert.That(Utils.array_allclose(expected, y.Data.Value));
            }

            [Test]
            public void Test_Forward2()
            {
                int n = 1, c = 5, h = 15, w = 15;
                int o = 8;
                (int, int) k = (3, 3), s = (3, 1), p = (2, 1);
                var x = xp.random.randn(n, c, h, w).astype("f").ToVariable();
                var W = xp.random.randn(o, c, k.Item1, k.Item2).astype("f").ToVariable();
                Variable b = null;
                var y = Utils.conv2d_simple(x, W, b, s, p);
                var expected = Chainer.CF.convolution_2d(x.Data.Value, W.Data.Value, b?.Data.Value, s, p);
                Assert.That(Utils.array_allclose(expected, y.Data.Value));
            }

            [Test]
            public void Test_Forward3()
            {
                int n = 1, c = 5, h = 20, w = 15;
                int o = 3;
                (int, int) k = (5, 3);
                int s = 1, p = 3;
                var x = xp.random.randn(n, c, h, w).astype("f").ToVariable();
                var W = xp.random.randn(o, c, k.Item1, k.Item2).astype("f").ToVariable();
                Variable b = null;
                var y = Utils.conv2d_simple(x, W, b, s, p);
                var expected = Chainer.CF.convolution_2d(x.Data.Value, W.Data.Value, b?.Data.Value, s, p);
                Assert.That(Utils.array_allclose(expected, y.Data.Value));
            }

            [Test]
            public void Test_Forward4()
            {
                int n = 1, c = 5, h = 20, w = 15;
                int o = 3;
                (int, int) k = (5, 3);
                int s = 1, p = 3;
                var x = xp.random.randn(n, c, h, w).astype("f").ToVariable();
                var W = xp.random.randn(o, c, k.Item1, k.Item2).astype("f").ToVariable();
                var b = xp.random.randn(o).astype("f").ToVariable();
                var y = Utils.conv2d_simple(x, W, b, s, p);
                var expected = Chainer.CF.convolution_2d(x.Data.Value, W.Data.Value, b.Data.Value, s, p);
                Assert.That(Utils.array_allclose(expected, y.Data.Value));
            }

            [Test]
            public void Test_Backward1()
            {
                int n = 1, c = 5, h = 20, w = 15;
                int o = 3;
                (int, int) k = (5, 3);
                int s = 1, p = 3;
                var x = xp.random.randn(n, c, h, w).ToVariable();
                var W = xp.random.randn(o, c, k.Item1, k.Item2).ToVariable();
                var b = xp.random.randn(o).ToVariable();
                var f = new Func<Params, Variable[]>(args => [Utils.conv2d_simple(args.Get<Variable>("x"), W, b, s, p)]);
                Assert.That(Utils.gradient_check(new Function(f), Params.New.SetKeywordArg(x, "x")));
            }

            [Test]
            public void Test_Backward2()
            {
                int n = 1, c = 5, h = 20, w = 15;
                int o = 3;
                (int, int) k = (5, 3);
                int s = 1, p = 3;
                var x = xp.random.randn(n, c, h, w).ToVariable();
                var W = xp.random.randn(o, c, k.Item1, k.Item2).ToVariable();
                var b = xp.random.randn(o).ToVariable();
                var f = new Func<Params, Variable[]>(args => [Utils.conv2d_simple(x, W, args.Get<Variable>("x"), s, p)]);
                Assert.That(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(b, arg1Name:"b")));
            }

            [Test]
            public void Test_Backward3()
            {
                int n = 1, c = 5, h = 20, w = 15;
                int o = 3;
                (int, int) k = (5, 3);
                int s = 1, p = 3;
                var x = xp.random.randn(n, c, h, w).ToVariable();
                var W = xp.random.randn(o, c, k.Item1, k.Item2).ToVariable();
                var b = xp.random.randn(o).ToVariable();
                var f = new Func<Params, Variable[]>(args => [Utils.conv2d_simple(x, args.Get<Variable>("x"), b, s, p)]);
                Assert.That(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(W, arg1Name: "W")));
            }
        }

        [Category("numpy")]
        public class np
        {
            [OneTimeSetUp]
            public void OneTimeSetUp()
            {
                if (string.IsNullOrEmpty(Runtime.PythonDLL))
                {
                    Runtime.PythonDLL = @"C:\Users\boiler\AppData\Local\Programs\Python\Python311\python311.dll";
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
                var x = xp.random.randn(n, c, h, w).astype("f").ToVariable();
                var W = xp.random.randn(o, c, k.Item1, k.Item2).astype("f").ToVariable();
                Variable b = null;
                var y = Utils.conv2d_simple(x, W, b, s, p);
                var expected = Chainer.CF.convolution_2d(x.Data.Value, W.Data.Value, b?.Data.Value, s, p);
                Assert.That(Utils.array_allclose(expected, y.Data.Value));
            }

            [Test]
            public void Test_Forward2()
            {
                int n = 1, c = 5, h = 15, w = 15;
                int o = 8;
                (int, int) k = (3, 3), s = (3, 1), p = (2, 1);
                var x = xp.random.randn(n, c, h, w).astype("f").ToVariable();
                var W = xp.random.randn(o, c, k.Item1, k.Item2).astype("f").ToVariable();
                Variable b = null;
                var y = Utils.conv2d_simple(x, W, b, s, p);
                var expected = Chainer.CF.convolution_2d(x.Data.Value, W.Data.Value, b?.Data.Value, s, p);
                Assert.That(Utils.array_allclose(expected, y.Data.Value));
            }

            [Test]
            public void Test_Forward3()
            {
                int n = 1, c = 5, h = 20, w = 15;
                int o = 3;
                (int, int) k = (5, 3);
                int s = 1, p = 3;
                var x = xp.random.randn(n, c, h, w).astype("f").ToVariable();
                var W = xp.random.randn(o, c, k.Item1, k.Item2).astype("f").ToVariable();
                Variable b = null;
                var y = Utils.conv2d_simple(x, W, b, s, p);
                var expected = Chainer.CF.convolution_2d(x.Data.Value, W.Data.Value, b?.Data.Value, s, p);
                Assert.That(Utils.array_allclose(expected, y.Data.Value));
            }

            [Test]
            public void Test_Forward4()
            {
                int n = 1, c = 5, h = 20, w = 15;
                int o = 3;
                (int, int) k = (5, 3);
                int s = 1, p = 3;
                var x = xp.random.randn(n, c, h, w).astype("f").ToVariable();
                var W = xp.random.randn(o, c, k.Item1, k.Item2).astype("f").ToVariable();
                var b = xp.random.randn(o).astype("f").ToVariable();
                var y = Utils.conv2d_simple(x, W, b, s, p);
                var expected = Chainer.CF.convolution_2d(x.Data.Value, W.Data.Value, b.Data.Value, s, p);
                Assert.That(Utils.array_allclose(expected, y.Data.Value));
            }

            [Test]
            public void Test_Backward1()
            {
                int n = 1, c = 5, h = 20, w = 15;
                int o = 3;
                (int, int) k = (5, 3);
                int s = 1, p = 3;
                var x = xp.random.randn(n, c, h, w).ToVariable();
                var W = xp.random.randn(o, c, k.Item1, k.Item2).ToVariable();
                var b = xp.random.randn(o).ToVariable();
                var f = new Func<Params, Variable[]>(args => [Utils.conv2d_simple(args.Get<Variable>("x"), W, b, s, p)]);
                Assert.That(Utils.gradient_check(new Function(f), Params.New.SetKeywordArg(x, "x")));
            }

            [Test]
            public void Test_Backward2()
            {
                int n = 1, c = 5, h = 20, w = 15;
                int o = 3;
                (int, int) k = (5, 3);
                int s = 1, p = 3;
                var x = xp.random.randn(n, c, h, w).ToVariable();
                var W = xp.random.randn(o, c, k.Item1, k.Item2).ToVariable();
                var b = xp.random.randn(o).ToVariable();
                var f = new Func<Params, Variable[]>(args => [Utils.conv2d_simple(x, W, args.Get<Variable>("x"), s, p)]);
                Assert.That(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(b, arg1Name: "b")));
            }

            [Test]
            public void Test_Backward3()
            {
                int n = 1, c = 5, h = 20, w = 15;
                int o = 3;
                (int, int) k = (5, 3);
                int s = 1, p = 3;
                var x = xp.random.randn(n, c, h, w).ToVariable();
                var W = xp.random.randn(o, c, k.Item1, k.Item2).ToVariable();
                var b = xp.random.randn(o).ToVariable();
                var f = new Func<Params, Variable[]>(args => [Utils.conv2d_simple(x, args.Get<Variable>("x"), b, s, p)]);
                Assert.That(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(W, arg1Name: "W")));
            }
        }
    }

    public class Conv2dTests
    {
        [Category("cupy")]
        public class cp
        {
            [OneTimeSetUp]
            public void OneTimeSetUp()
            {
                if (string.IsNullOrEmpty(Runtime.PythonDLL))
                {
                    Runtime.PythonDLL = @"C:\Users\boiler\AppData\Local\Programs\Python\Python311\python311.dll";
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
                var x = xp.random.randn(n, c, h, w).astype("f").ToVariable();
                var W = xp.random.randn(o, c, k.Item1, k.Item2).astype("f").ToVariable();
                Variable b = null;
                var y = Conv2d.Invoke(x, W, b, s, p);
                var expected = Chainer.CF.convolution_2d(x.Data.Value, W.Data.Value, b?.Data.Value, s, p);
                Assert.That(Utils.array_allclose(expected, y[0].Data.Value));
            }

            [Test]
            public void Test_Forward2()
            {
                int n = 1, c = 5, h = 15, w = 15;
                int o = 8;
                (int, int) k = (3, 3), s = (3, 1), p = (2, 1);
                var x = xp.random.randn(n, c, h, w).astype("f").ToVariable();
                var W = xp.random.randn(o, c, k.Item1, k.Item2).astype("f").ToVariable();
                Variable b = null;
                var y = Conv2d.Invoke(x, W, b, s, p);
                var expected = Chainer.CF.convolution_2d(x.Data.Value, W.Data.Value, b?.Data.Value, s, p);
                Assert.That(Utils.array_allclose(expected, y[0].Data.Value));
            }

            [Test]
            public void Test_Forward3()
            {
                int n = 1, c = 5, h = 20, w = 15;
                int o = 3;
                (int, int) k = (5, 3);
                int s = 1, p = 3;
                var x = xp.random.randn(n, c, h, w).astype("f").ToVariable();
                var W = xp.random.randn(o, c, k.Item1, k.Item2).astype("f").ToVariable();
                Variable b = null;
                var y = Conv2d.Invoke(x, W, b, s, p);
                var expected = Chainer.CF.convolution_2d(x.Data.Value, W.Data.Value, b?.Data.Value, s, p);
                Assert.That(Utils.array_allclose(expected, y[0].Data.Value));
            }

            [Test]
            public void Test_Forward4()
            {
                int n = 1, c = 5, h = 20, w = 15;
                int o = 3;
                (int, int) k = (5, 3);
                int s = 1, p = 3;
                var x = xp.random.randn(n, c, h, w).astype("f").ToVariable();
                var W = xp.random.randn(o, c, k.Item1, k.Item2).astype("f").ToVariable();
                var b = xp.random.randn(o).astype("f").ToVariable();
                var y = Conv2d.Invoke(x, W, b, s, p);
                var expected = Chainer.CF.convolution_2d(x.Data.Value, W.Data.Value, b?.Data.Value, s, p);
                Assert.That(Utils.array_allclose(expected, y[0].Data.Value));
            }

            [Test]
            public void Test_Backward1()
            {
                int n = 1, c = 5, h = 20, w = 15;
                int o = 3;
                (int, int) k = (5, 3);
                int s = 1, p = 3;
                var x = xp.random.randn(n, c, h, w).ToVariable();
                var W = xp.random.randn(o, c, k.Item1, k.Item2).ToVariable();
                var b = xp.random.randn(o)?.ToVariable();
                var f = new Func<Params, Variable[]>(args => Conv2d.Invoke(args.Get<Variable>("x"), W, b, s, p));
                Assert.That(Utils.gradient_check(new Function(f), Params.New.SetKeywordArg(x, "x")));
            }

            [Test]
            public void Test_Backward2()
            {
                int n = 1, c = 5, h = 20, w = 15;
                int o = 3;
                (int, int) k = (5, 3);
                int s = 1, p = 3;
                var x = xp.random.randn(n, c, h, w).ToVariable();
                var W = xp.random.randn(o, c, k.Item1, k.Item2).ToVariable();
                var b = xp.random.randn(o).ToVariable();
                var f = new Func<Params, Variable[]>(args => Conv2d.Invoke(x, W, args.Get<Variable>("x"), s, p));
                Assert.That(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(b, arg1Name: "b")));
            }

            [Test]
            public void Test_Backward3()
            {
                int n = 1, c = 5, h = 20, w = 15;
                int o = 3;
                (int, int) k = (5, 3);
                int s = 1, p = 3;
                var x = xp.random.randn(n, c, h, w).ToVariable();
                var W = xp.random.randn(o, c, k.Item1, k.Item2).ToVariable();
                var b = xp.random.randn(o)?.ToVariable();
                var f = new Func<Params, Variable[]>(args => Conv2d.Invoke(x, args.Get<Variable>("x"), b, s, p));
                Assert.That(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(W, arg1Name: "W")));
            }
        }

        [Category("numpy")]
        public class np
        {
            [OneTimeSetUp]
            public void OneTimeSetUp()
            {
                if (string.IsNullOrEmpty(Runtime.PythonDLL))
                {
                    Runtime.PythonDLL = @"C:\Users\boiler\AppData\Local\Programs\Python\Python311\python311.dll";
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
                var x = xp.random.randn(n, c, h, w).astype("f").ToVariable();
                var W = xp.random.randn(o, c, k.Item1, k.Item2).astype("f").ToVariable();
                Variable b = null;
                var y = Conv2d.Invoke(x, W, b, s, p);
                var expected = Chainer.CF.convolution_2d(x.Data.Value, W.Data.Value, b?.Data.Value, s, p);
                Assert.That(Utils.array_allclose(expected, y[0].Data.Value));
            }

            [Test]
            public void Test_Forward2()
            {
                int n = 1, c = 5, h = 15, w = 15;
                int o = 8;
                (int, int) k = (3, 3), s = (3, 1), p = (2, 1);
                var x = xp.random.randn(n, c, h, w).astype("f").ToVariable();
                var W = xp.random.randn(o, c, k.Item1, k.Item2).astype("f").ToVariable();
                Variable b = null;
                var y = Conv2d.Invoke(x, W, b, s, p);
                var expected = Chainer.CF.convolution_2d(x.Data.Value, W.Data.Value, b?.Data.Value, s, p);
                Assert.That(Utils.array_allclose(expected, y[0].Data.Value));
            }

            [Test]
            public void Test_Forward3()
            {
                int n = 1, c = 5, h = 20, w = 15;
                int o = 3;
                (int, int) k = (5, 3);
                int s = 1, p = 3;
                var x = xp.random.randn(n, c, h, w).astype("f").ToVariable();
                var W = xp.random.randn(o, c, k.Item1, k.Item2).astype("f").ToVariable();
                Variable b = null;
                var y = Conv2d.Invoke(x, W, b, s, p);
                var expected = Chainer.CF.convolution_2d(x.Data.Value, W.Data.Value, b?.Data.Value, s, p);
                Assert.That(Utils.array_allclose(expected, y[0].Data.Value));
            }

            [Test]
            public void Test_Forward4()
            {
                int n = 1, c = 5, h = 20, w = 15;
                int o = 3;
                (int, int) k = (5, 3);
                int s = 1, p = 3;
                var x = xp.random.randn(n, c, h, w).astype("f").ToVariable();
                var W = xp.random.randn(o, c, k.Item1, k.Item2).astype("f").ToVariable();
                var b = xp.random.randn(o).astype("f").ToVariable();
                var y = Conv2d.Invoke(x, W, b, s, p);
                var expected = Chainer.CF.convolution_2d(x.Data.Value, W.Data.Value, b?.Data.Value, s, p);
                Assert.That(Utils.array_allclose(expected, y[0].Data.Value));
            }

            [Test]
            public void Test_Backward1()
            {
                int n = 1, c = 5, h = 20, w = 15;
                int o = 3;
                (int, int) k = (5, 3);
                int s = 1, p = 3;
                var x = xp.random.randn(n, c, h, w).ToVariable();
                var W = xp.random.randn(o, c, k.Item1, k.Item2).ToVariable();
                var b = xp.random.randn(o)?.ToVariable();
                var f = new Func<Params, Variable[]>(args => Conv2d.Invoke(args.Get<Variable>("x"), W, b, s, p));
                Assert.That(Utils.gradient_check(new Function(f), Params.New.SetKeywordArg(x, "x")));
            }

            [Test]
            public void Test_Backward2()
            {
                int n = 1, c = 5, h = 20, w = 15;
                int o = 3;
                (int, int) k = (5, 3);
                int s = 1, p = 3;
                var x = xp.random.randn(n, c, h, w).ToVariable();
                var W = xp.random.randn(o, c, k.Item1, k.Item2).ToVariable();
                var b = xp.random.randn(o).ToVariable();
                var f = new Func<Params, Variable[]>(args => Conv2d.Invoke(x, W, args.Get<Variable>("x"), s, p));
                Assert.That(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(b, arg1Name: "b")));
            }

            [Test]
            public void Test_Backward3()
            {
                int n = 1, c = 5, h = 20, w = 15;
                int o = 3;
                (int, int) k = (5, 3);
                int s = 1, p = 3;
                var x = xp.random.randn(n, c, h, w).ToVariable();
                var W = xp.random.randn(o, c, k.Item1, k.Item2).ToVariable();
                var b = xp.random.randn(o)?.ToVariable();
                var f = new Func<Params, Variable[]>(args => Conv2d.Invoke(x, args.Get<Variable>("x"), b, s, p));
                Assert.That(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(W, arg1Name: "W")));
            }
        }
    }
}
