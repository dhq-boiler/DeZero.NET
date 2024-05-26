using DeZero.NET.Core;
using DeZero.NET.Functions;
using DeZero.NET.Tests.Chainer;
using Python.Runtime;

namespace DeZero.NET.Tests
{
    public class SoftmaxCrossEntropyTests
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
                var x = xp.array([[-1, 0, 1, 2], [2, 0, 1, -1]], xp.float32).ToVariable();
                var t = xp.array([3, 0]).astype(xp.int32).ToVariable();
                var y = SoftmaxCrossEntropy.Invoke(x, t)[0];
                var y2 = CF.softmax_cross_entropy(x.Data.Value, t.Data.Value);
                Assert.IsTrue(Utils.array_allclose(y.Data.Value, y2));
            }

            [Test]
            public void Test_Backward1()
            {
                var x = xp.array([[-1, 0, 1, 2], [2, 0, 1, -1]], xp.float32).ToVariable();
                var t = xp.array([3, 0]).astype(xp.int32).ToVariable();
                Func<Params, Variable[]> f = args => SoftmaxCrossEntropy.Invoke(args.Get<Variable>("x"), t);
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x)));
            }

            [Test]
            public void Test_Backward2()
            {
                var (N, CLS_NUM) = (10, 10);
                var x = xp.random.randn(N, CLS_NUM).ToVariable();
                var t = xp.random.randint(0, CLS_NUM, [N]).ToVariable();
                Func<Params, Variable[]> f = args => SoftmaxCrossEntropy.Invoke(args.Get<Variable>("x"), t);
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x)));
            }

            [Test]
            public void Test_Backward3()
            {
                var (N, CLS_NUM) = (100, 10);
                var x = xp.random.randn(N, CLS_NUM).ToVariable();
                var t = xp.random.randint(0, CLS_NUM, [N]).ToVariable();
                Func<Params, Variable[]> f = args => SoftmaxCrossEntropy.Invoke(args.Get<Variable>("x"), t);
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
                var x = xp.array([[-1, 0, 1, 2], [2, 0, 1, -1]], xp.float32).ToVariable();
                var t = xp.array([3, 0]).astype(xp.int32).ToVariable();
                var y = SoftmaxCrossEntropy.Invoke(x, t)[0];
                var y2 = CF.softmax_cross_entropy(x.Data.Value, t.Data.Value);
                Assert.IsTrue(Utils.array_allclose(y.Data.Value, y2));
            }

            [Test]
            public void Test_Backward1()
            {
                var x = xp.array([[-1, 0, 1, 2], [2, 0, 1, -1]], xp.float32).ToVariable();
                var t = xp.array([3, 0]).astype(xp.int32).ToVariable();
                Func<Params, Variable[]> f = args => SoftmaxCrossEntropy.Invoke(args.Get<Variable>("x"), t);
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x)));
            }

            [Test]
            public void Test_Backward2()
            {
                var (N, CLS_NUM) = (10, 10);
                var x = xp.random.randn(N, CLS_NUM).ToVariable();
                var t = xp.random.randint(0, CLS_NUM, [N]).ToVariable();
                Func<Params, Variable[]> f = args => SoftmaxCrossEntropy.Invoke(args.Get<Variable>("x"), t);
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x)));
            }

            [Test]
            public void Test_Backward3()
            {
                var (N, CLS_NUM) = (100, 10);
                var x = xp.random.randn(N, CLS_NUM).ToVariable();
                var t = xp.random.randint(0, CLS_NUM, [N]).ToVariable();
                Func<Params, Variable[]> f = args => SoftmaxCrossEntropy.Invoke(args.Get<Variable>("x"), t);
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x)));
            }
        }
    }

    public class SoftmaxCrossEntropy_simple_Tests
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
                var x = xp.array([[-1, 0, 1, 2], [2, 0, 1, -1]], xp.float32).ToVariable();
                var t = xp.array([3, 0]).astype(xp.int32).ToVariable();
                var y = Loss.SoftmaxCrossEntropy_simple(x, t);
                var y2 = CF.softmax_cross_entropy(x.Data.Value, t.Data.Value);
                Assert.IsTrue(Utils.array_allclose(y.Data.Value, y2));
            }

            [Test]
            public void Test_Backward1()
            {
                var x = xp.array([[-1, 0, 1, 2], [2, 0, 1, -1]], xp.float32).ToVariable();
                var t = xp.array([3, 0]).astype(xp.int32).ToVariable();
                Func<Params, Variable[]> f = args => [Loss.SoftmaxCrossEntropy_simple(args.Get<Variable>("x"), t)];
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x)));
            }

            [Test]
            public void Test_Backward2()
            {
                var (N, CLS_NUM) = (10, 10);
                xp.random.seed(0);
                var x = xp.random.randn(N, CLS_NUM).astype(xp.int32).ToVariable();
                var t = xp.random.randint(0, CLS_NUM, [N]).astype(xp.int32).ToVariable();
                Func<Params, Variable[]> f = args => [Loss.SoftmaxCrossEntropy_simple(args.Get<Variable>("x"), t)];
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x)));
            }

            [Test]
            public void Test_Backward3()
            {
                var (N, CLS_NUM) = (100, 10);
                var x = xp.random.randn(N, CLS_NUM).ToVariable();
                var t = xp.random.randint(0, CLS_NUM, [N]).ToVariable();
                Func<Params, Variable[]> f = args => [Loss.SoftmaxCrossEntropy_simple(args.Get<Variable>("x"), t)];
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
                var x = xp.array([[-1, 0, 1, 2], [2, 0, 1, -1]], xp.float32).ToVariable();
                var t = xp.array([3, 0]).astype(xp.int32).ToVariable();
                var y = Loss.SoftmaxCrossEntropy_simple(x, t);
                var y2 = CF.softmax_cross_entropy(x.Data.Value, t.Data.Value);
                Assert.IsTrue(Utils.array_allclose(y.Data.Value, y2));
            }

            [Test]
            public void Test_Backward1()
            {
                var x = xp.array([[-1, 0, 1, 2], [2, 0, 1, -1]], xp.float32).ToVariable();
                var t = xp.array([3, 0]).astype(xp.int32).ToVariable();
                Func<Params, Variable[]> f = args => [Loss.SoftmaxCrossEntropy_simple(args.Get<Variable>("x"), t)];
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x)));
            }

            [Test]
            public void Test_Backward2()
            {
                var (N, CLS_NUM) = (10, 10);
                var x = xp.random.randn(N, CLS_NUM).ToVariable();
                var t = xp.random.randint(0, CLS_NUM, [N]).ToVariable();
                Func<Params, Variable[]> f = args => [Loss.SoftmaxCrossEntropy_simple(args.Get<Variable>("x"), t)];
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x)));
            }

            [Test]
            public void Test_Backward3()
            {
                var (N, CLS_NUM) = (100, 10);
                var x = xp.random.randn(N, CLS_NUM).ToVariable();
                var t = xp.random.randint(0, CLS_NUM, [N]).ToVariable();
                Func<Params, Variable[]> f = args => [Loss.SoftmaxCrossEntropy_simple(args.Get<Variable>("x"), t)];
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x)));
            }
        }
    }
}
