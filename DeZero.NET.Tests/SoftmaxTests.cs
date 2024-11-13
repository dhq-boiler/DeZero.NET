using DeZero.NET.Core;
using DeZero.NET.Extensions;
using DeZero.NET.Functions;
using DeZero.NET.Tests.Chainer;
using Python.Runtime;

namespace DeZero.NET.Tests
{
    public class SoftmaxTests
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
                var x = xp.array([[0, 1, 2], [0, 2, 4]], xp.float32).ToVariable();
                var y2 = CF.softmax(x.Data.Value, axis: 1);
                var y = Softmax.Invoke(x)[0];
                Assert.That(Utils.array_allclose(y.Data.Value, y2));
            }

            [Test]
            public void Test_Forward2()
            {
                xp.random.seed(0);
                var x = xp.random.rand(10, 10).astype("f").ToVariable();
                var y2 = CF.softmax(x.Data.Value, axis: 1);
                var y = Softmax.Invoke(x)[0];
                Assert.That(Utils.array_allclose(y.Data.Value, y2));
            }

            [Test]
            public void Test_Forward3()
            {
                xp.random.seed(0);
                var x = xp.random.rand(10, 10, 10).astype("f").ToVariable();
                var y2 = CF.softmax(x.Data.Value, axis: 1);
                var y = Softmax.Invoke(x)[0];
                Assert.That(Utils.array_allclose(y.Data.Value, y2));
            }

            [Test]
            public void Test_Backward1()
            {
                var x_data = xp.array([[0, 1, 2], [0, 2, 4]]).ToVariable();
                Func<Params, Variable[]> f = args => Softmax.Invoke(args.Get<Variable>("x"), axis: [1]);
                Assert.That(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x_data)));
            }

            [Test]
            public void Test_Backward2()
            {
                xp.random.seed(0);
                var x_data = xp.random.rand(10, 10).ToVariable();
                Func<Params, Variable[]> f = args => Softmax.Invoke(args.Get<Variable>("x"), axis: [1]);
                Assert.That(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x_data)));
            }

            [Test]
            public void Test_Backward3()
            {
                xp.random.seed(0);
                var x_data = xp.random.rand(10, 10, 10).ToVariable();
                Func<Params, Variable[]> f = args => Softmax.Invoke(args.Get<Variable>("x"), axis: [1]);
                Assert.That(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x_data)));
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
                var x = xp.array([[0, 1, 2], [0, 2, 4]], xp.float32).ToVariable();
                var y2 = CF.softmax(x.Data.Value, axis: 1);
                var y = Softmax.Invoke(x)[0];
                Assert.That(Utils.array_allclose(y.Data.Value, y2));
            }

            [Test]
            public void Test_Forward2()
            {
                xp.random.seed(0);
                var x = xp.random.rand(10, 10).astype("f").ToVariable();
                var y2 = CF.softmax(x.Data.Value, axis: 1);
                var y = Softmax.Invoke(x)[0];
                Assert.That(Utils.array_allclose(y.Data.Value, y2));
            }

            [Test]
            public void Test_Forward3()
            {
                xp.random.seed(0);
                var x = xp.random.rand(10, 10, 10).astype("f").ToVariable();
                var y2 = CF.softmax(x.Data.Value, axis: 1);
                var y = Softmax.Invoke(x)[0];
                Assert.That(Utils.array_allclose(y.Data.Value, y2));
            }

            [Test]
            public void Test_Backward1()
            {
                var x_data = xp.array([[0, 1, 2], [0, 2, 4]]).ToVariable();
                Func<Params, Variable[]> f = args => Softmax.Invoke(args.Get<Variable>("x"), axis: [1]);
                Assert.That(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x_data)));
            }

            [Test]
            public void Test_Backward2()
            {
                xp.random.seed(0);
                var x_data = xp.random.rand(10, 10).ToVariable();
                Func<Params, Variable[]> f = args => Softmax.Invoke(args.Get<Variable>("x"), axis: [1]);
                Assert.That(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x_data)));
            }

            [Test]
            public void Test_Backward3()
            {
                xp.random.seed(0);
                var x_data = xp.random.rand(10, 10, 10).ToVariable();
                Func<Params, Variable[]> f = args => Softmax.Invoke(args.Get<Variable>("x"), axis: [1]);
                Assert.That(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x_data)));
            }
        }
    }

    public class Softmax_simple_Tests
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
                var x = xp.array([[0, 1, 2], [0, 2, 4]], xp.float32).ToVariable();
                var y2 = CF.softmax(x.Data.Value, axis: 1);
                var y = Loss.Softmax_simple(x);
                Assert.That(Utils.array_allclose(y.Data.Value, y2));
            }

            [Test]
            public void Test_Forward2()
            {
                xp.random.seed(0);
                var x = xp.random.rand(10, 10).astype("f").ToVariable();
                var y2 = CF.softmax(x.Data.Value, axis: 1);
                var y = Loss.Softmax_simple(x);
                Assert.That(Utils.array_allclose(y.Data.Value, y2));
            }

            [Test]
            public void Test_Forward3()
            {
                xp.random.seed(0);
                var x = xp.random.rand(10, 10, 10).astype("f").ToVariable();
                var y2 = CF.softmax(x.Data.Value, axis: 1);
                var y = Loss.Softmax_simple(x);
                Assert.That(Utils.array_allclose(y.Data.Value, y2));
            }

            [Test]
            public void Test_Backward1()
            {
                var x_data = xp.array([[0, 1, 2], [0, 2, 4]]).ToVariable();
                Func<Params, Variable[]> f = args => [Loss.Softmax_simple(args.Get<Variable>("x"), axis: [1])];
                Assert.That(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x_data)));
            }

            [Test]
            public void Test_Backward2()
            {
                xp.random.seed(0);
                var x_data = xp.random.rand(10, 10).ToVariable();
                Func<Params, Variable[]> f = args => [Loss.Softmax_simple(args.Get<Variable>("x"), axis: [1])];
                Assert.That(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x_data)));
            }

            [Test]
            public void Test_Backward3()
            {
                xp.random.seed(0);
                var x_data = xp.random.rand(10, 10, 10).ToVariable();
                Func<Params, Variable[]> f = args => [Loss.Softmax_simple(args.Get<Variable>("x"), axis: [1])];
                Assert.That(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x_data)));
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
                var x = xp.array([[0, 1, 2], [0, 2, 4]], xp.float32).ToVariable();
                var y2 = CF.softmax(x.Data.Value, axis: 1);
                var y = Loss.Softmax_simple(x);
                Assert.That(Utils.array_allclose(y.Data.Value, y2));
            }

            [Test]
            public void Test_Forward2()
            {
                xp.random.seed(0);
                var x = xp.random.rand(10, 10).astype("f").ToVariable();
                var y2 = CF.softmax(x.Data.Value, axis: 1);
                var y = Loss.Softmax_simple(x);
                Assert.That(Utils.array_allclose(y.Data.Value, y2));
            }

            [Test]
            public void Test_Forward3()
            {
                xp.random.seed(0);
                var x = xp.random.rand(10, 10, 10).astype("f").ToVariable();
                var y2 = CF.softmax(x.Data.Value, axis: 1);
                var y = Loss.Softmax_simple(x);
                Assert.That(Utils.array_allclose(y.Data.Value, y2));
            }

            [Test]
            public void Test_Backward1()
            {
                var x_data = xp.array([[0, 1, 2], [0, 2, 4]]).ToVariable();
                Func<Params, Variable[]> f = args => [Loss.Softmax_simple(args.Get<Variable>("x"), axis: [1])];
                Assert.That(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x_data)));
            }

            [Test]
            public void Test_Backward2()
            {
                xp.random.seed(0);
                var x_data = xp.random.rand(10, 10).ToVariable();
                Func<Params, Variable[]> f = args => [Loss.Softmax_simple(args.Get<Variable>("x"), axis: [1])];
                Assert.That(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x_data)));
            }

            [Test]
            public void Test_Backward3()
            {
                xp.random.seed(0);
                var x_data = xp.random.rand(10, 10, 10).ToVariable();
                Func<Params, Variable[]> f = args => [Loss.Softmax_simple(args.Get<Variable>("x"), axis: [1])];
                Assert.That(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x_data)));
            }
        }
    }

    public class LogSoftmaxTests
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
                var x = xp.array([[-1, 0, 1, 2], [2, 0, 1, -1]], xp.float32).ToVariable();
                var y = LogSoftmax.Invoke(x)[0];
                var y2 = CF.log_softmax(x.Data.Value, axis: 1);
                Assert.That(Utils.array_allclose(y.Data.Value, y2));
            }

            [Test]
            public void Test_Backward1()
            {
                var x = xp.array([[-1, 0, 1, 2], [2, 0, 1, -1]]).ToVariable();
                Func<Params, Variable[]> f = args => LogSoftmax.Invoke(args.Get<Variable>("x"), axis: [1]);
                Assert.That(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x)));
            }

            [Test]
            public void Test_Backward2()
            {
                var x = xp.random.randn(10, 10).ToVariable();
                Func<Params, Variable[]> f = args => LogSoftmax.Invoke(args.Get<Variable>("x"), axis: [1]);
                Assert.That(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x)));
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
                var x = xp.array([[-1, 0, 1, 2], [2, 0, 1, -1]], xp.float32).ToVariable();
                var y = LogSoftmax.Invoke(x)[0];
                var y2 = CF.log_softmax(x.Data.Value, axis: 1);
                Assert.That(Utils.array_allclose(y.Data.Value, y2));
            }

            [Test]
            public void Test_Backward1()
            {
                var x = xp.array([[-1, 0, 1, 2], [2, 0, 1, -1]]).ToVariable();
                Func<Params, Variable[]> f = args => LogSoftmax.Invoke(args.Get<Variable>("x"), axis: [1]);
                Assert.That(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x)));
            }

            [Test]
            public void Test_Backward2()
            {
                var x = xp.random.randn(10, 10).ToVariable();
                Func<Params, Variable[]> f = args => LogSoftmax.Invoke(args.Get<Variable>("x"), axis: [1]);
                Assert.That(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x)));
            }
        }
    }
}
