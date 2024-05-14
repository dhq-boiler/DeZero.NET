using DeZero.NET.Core;
using DeZero.NET.Functions;
using DeZero.NET.Tests.Chainer;
using Python.Runtime;

namespace DeZero.NET.Tests
{
    public class PoolingTests
    {
        [Category("cupy")]
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
                int n = 1, c = 5, h = 16, w = 16;
                int ksize = 2, stride = 2, pad = 0;
                var x = xp.random.randn(n, c, h, w).astype("f").ToVariable();

                var y = Pooling.Invoke(x, (ksize, ksize), stride, pad)[0];
                var expected = CF.max_pooling_2d(x.Data, ksize, stride, pad);
                Assert.IsTrue(Utils.array_allclose(expected, y.Data));
            }

            [Test]
            public void Test_Forward2()
            {
                int n = 1, c = 5, h = 15, w = 15;
                int ksize = 2, stride = 2, pad = 0;
                var x = xp.random.randn(n, c, h, w).astype("f").ToVariable();

                var y = Pooling.Invoke(x, (ksize, ksize), stride, pad)[0];
                var expected = CF.max_pooling_2d(x.Data, ksize, stride, pad, cover_all:false);
                Assert.IsTrue(Utils.array_allclose(expected, y.Data));
            }

            [Test]
            public void Test_Backward1()
            {
                int n = 1, c = 5, h = 16, w = 16;
                int ksize = 2, stride = 2, pad = 0;
                var x = (xp.random.randn(n, c, h, w).astype("f") * 1000).ToVariable();
                Func<Params, Variable[]> f = args => [Pooling.Invoke(x, (ksize, ksize), stride, pad)[0]];
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x)));
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
                int n = 1, c = 5, h = 16, w = 16;
                int ksize = 2, stride = 2, pad = 0;
                var x = xp.random.randn(n, c, h, w).astype("f").ToVariable();

                var y = Pooling.Invoke(x, (ksize, ksize), stride, pad)[0];
                var expected = CF.max_pooling_2d(x.Data, ksize, stride, pad);
                Assert.IsTrue(Utils.array_allclose(expected, y.Data));
            }

            [Test]
            public void Test_Forward2()
            {
                int n = 1, c = 5, h = 15, w = 15;
                int ksize = 2, stride = 2, pad = 0;
                var x = xp.random.randn(n, c, h, w).astype("f").ToVariable();

                var y = Pooling.Invoke(x, (ksize, ksize), stride, pad)[0];
                var expected = CF.max_pooling_2d(x.Data, ksize, stride, pad, cover_all: false);
                Assert.IsTrue(Utils.array_allclose(expected, y.Data));
            }

            [Test]
            public void Test_Backward1()
            {
                int n = 1, c = 5, h = 16, w = 16;
                int ksize = 2, stride = 2, pad = 0;
                var x = (xp.random.randn(n, c, h, w).astype("f") * 1000).ToVariable();
                Func<Params, Variable[]> f = args => [Pooling.Invoke(x, (ksize, ksize), stride, pad)[0]];
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x)));
            }
        }
    }

    public class Pooling_simple_Tests
    {
        [Category("cupy")]
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
                int n = 1, c = 5, h = 16, w = 16;
                int ksize = 2, stride = 2, pad = 0;
                var x = xp.random.randn(n, c, h, w).astype("f").ToVariable();

                var y = Conv.pooling_simple(x, ksize, stride, pad)[0];
                var expected = CF.max_pooling_2d(x.Data, ksize, stride, pad);
                Assert.IsTrue(Utils.array_allclose(expected, y.Data));
            }

            [Test]
            public void Test_Forward2()
            {
                int n = 1, c = 5, h = 15, w = 15;
                int ksize = 2, stride = 2, pad = 0;
                var x = xp.random.randn(n, c, h, w).astype("f").ToVariable();

                var y = Conv.pooling_simple(x, ksize, stride, pad)[0];
                var expected = CF.max_pooling_2d(x.Data, ksize, stride, pad, cover_all: false);
                Assert.IsTrue(Utils.array_allclose(expected, y.Data));
            }

            [Test]
            public void Test_Backward1()
            {
                int n = 1, c = 5, h = 16, w = 16;
                int ksize = 2, stride = 2, pad = 0;
                var x = (xp.random.randn(n, c, h, w).astype("f") * 1000).ToVariable();
                Func<Params, Variable[]> f = args => [Conv.pooling_simple(x, ksize, stride, pad)];
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x)));
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
                int n = 1, c = 5, h = 16, w = 16;
                int ksize = 2, stride = 2, pad = 0;
                var x = xp.random.randn(n, c, h, w).astype("f").ToVariable();

                var y = Conv.pooling_simple(x, ksize, stride, pad)[0];
                var expected = CF.max_pooling_2d(x.Data, ksize, stride, pad);
                Assert.IsTrue(Utils.array_allclose(expected, y.Data));
            }

            [Test]
            public void Test_Forward2()
            {
                int n = 1, c = 5, h = 15, w = 15;
                int ksize = 2, stride = 2, pad = 0;
                var x = xp.random.randn(n, c, h, w).astype("f").ToVariable();

                var y = Conv.pooling_simple(x, ksize, stride, pad)[0];
                var expected = CF.max_pooling_2d(x.Data, ksize, stride, pad, cover_all: false);
                Assert.IsTrue(Utils.array_allclose(expected, y.Data));
            }

            [Test]
            public void Test_Backward1()
            {
                int n = 1, c = 5, h = 16, w = 16;
                int ksize = 2, stride = 2, pad = 0;
                var x = (xp.random.randn(n, c, h, w).astype("f") * 1000).ToVariable();
                Func<Params, Variable[]> f = args => [Conv.pooling_simple(x, ksize, stride, pad)];
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x)));
            }
        }
    }


    public class AveragePoolingTests
    {
        [Category("cupy")]
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
                int n = 1, c = 5, h = 16, w = 16;
                int ksize = 2, stride = 2, pad = 0;
                var x = xp.random.randn(n, c, h, w).astype("f").ToVariable();

                var y = AveragePooling.Invoke(x, (ksize, ksize), stride, pad)[0];
                var expected = CF.average_pooling_2d(x.Data, ksize, stride, pad);
                Assert.IsTrue(Utils.array_allclose(expected, y.Data));
            }

            [Test]
            public void Test_Forward2()
            {
                int n = 1, c = 5, h = 15, w = 15;
                int ksize = 2, stride = 2, pad = 0;
                var x = xp.random.randn(n, c, h, w).astype("f").ToVariable();

                var y = AveragePooling.Invoke(x, (ksize, ksize), stride, pad)[0];
                var expected = CF.average_pooling_2d(x.Data, ksize, stride, pad);
                Assert.IsTrue(Utils.array_allclose(expected, y.Data));
            }

            [Test]
            public void Test_Backward1()
            {
                int n = 1, c = 5, h = 16, w = 16;
                int ksize = 2, stride = 2, pad = 0;
                var x = (xp.random.randn(n, c, h, w).astype("f") * 1000).ToVariable();
                var ap = new AveragePooling((ksize, ksize), stride, pad);
                ap.F = args => [AveragePooling.Invoke(ap, args.Get<Variable>("x"))[0]];
                
                Assert.IsTrue(Utils.gradient_check(ap, Params.New.SetPositionalArgs(x)));
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
                int n = 1, c = 5, h = 16, w = 16;
                int ksize = 2, stride = 2, pad = 0;
                var x = xp.random.randn(n, c, h, w).astype("f").ToVariable();

                var y = AveragePooling.Invoke(x, (ksize, ksize), stride, pad)[0];
                var expected = CF.average_pooling_2d(x.Data, ksize, stride, pad);
                Assert.IsTrue(Utils.array_allclose(expected, y.Data));
            }

            [Test]
            public void Test_Forward2()
            {
                int n = 1, c = 5, h = 15, w = 15;
                int ksize = 2, stride = 2, pad = 0;
                var x = xp.random.randn(n, c, h, w).astype("f").ToVariable();

                var y = AveragePooling.Invoke(x, (ksize, ksize), stride, pad)[0];
                var expected = CF.average_pooling_2d(x.Data, ksize, stride, pad);
                Assert.IsTrue(Utils.array_allclose(expected, y.Data));
            }

            [Test]
            public void Test_Backward1()
            {
                int n = 1, c = 5, h = 16, w = 16;
                int ksize = 2, stride = 2, pad = 0;
                var x = (xp.random.randn(n, c, h, w).astype("f") * 1000).ToVariable();
                Func<Params, Variable[]> f = args => [AveragePooling.Invoke(args.Get<Variable>("x"), (ksize, ksize), stride, pad)[0]];
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x)));
            }
        }
    }
}
