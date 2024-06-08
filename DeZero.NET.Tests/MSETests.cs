using DeZero.NET.Core;
using DeZero.NET.Functions;
using Python.Runtime;

namespace DeZero.NET.Tests
{
    public class MSETests
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
                var x0 = xp.array([0.0, 1.0, 2.0]).ToVariable();
                var x1 = xp.array([0.0, 1.0, 2.0]).ToVariable();
                var expected = ((x0 - x1) * (x0 - x1)).Data.Value.sum() / x0.size;
                var y = MeanSquaredError.Invoke(x0, x1)[0];
                Assert.IsTrue(Utils.array_allclose(y.Data.Value, expected));
            }

            [Test]
            public void Test_Backward1()
            {
                var x0 = xp.random.rand(10).ToVariable();
                var x1 = xp.random.rand(10).ToVariable();
                Func<Params, Variable[]> f = args => [MeanSquaredError.Invoke(args.Get<Variable>("x"), x1)[0]];
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x0)));
            }

            [Test]
            public void Test_Backward2()
            {
                var x0 = xp.random.rand(100).ToVariable();
                var x1 = xp.random.rand(100).ToVariable();
                Func<Params, Variable[]> f = args => [MeanSquaredError.Invoke(args.Get<Variable>("x"), x1)[0]];
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x0)));
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
                var x0 = xp.array([0.0, 1.0, 2.0]).ToVariable();
                var x1 = xp.array([0.0, 1.0, 2.0]).ToVariable();
                var expected = ((x0 - x1) * (x0 - x1)).Data.Value.sum() / x0.size;
                var y = MeanSquaredError.Invoke(x0, x1)[0];
                Assert.IsTrue(Utils.array_allclose(y.Data.Value, expected));
            }

            [Test]
            public void Test_Backward1()
            {
                var x0 = xp.random.rand(10).ToVariable();
                var x1 = xp.random.rand(10).ToVariable();
                Func<Params, Variable[]> f = args => [MeanSquaredError.Invoke(args.Get<Variable>("x"), x1)[0]];
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x0)));
            }

            [Test]
            public void Test_Backward2()
            {
                var x0 = xp.random.rand(100).ToVariable();
                var x1 = xp.random.rand(100).ToVariable();
                Func<Params, Variable[]> f = args => [MeanSquaredError.Invoke(args.Get<Variable>("x"), x1)[0]];
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x0)));
            }
        }
    }

    public class MSE_simple_Tests
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
                var x0 = xp.array([0.0, 1.0, 2.0]).ToVariable();
                var x1 = xp.array([0.0, 1.0, 2.0]).ToVariable();
                var expected = ((x0 - x1) * (x0 - x1)).Data.Value.sum() / x0.size;
                var y = Loss.MeanSquaredError_simple(x0, x1);
                Assert.IsTrue(Utils.array_allclose(y.Data.Value, expected));
            }

            [Test]
            public void Test_Backward1()
            {
                var x0 = xp.random.rand(10).ToVariable();
                var x1 = xp.random.rand(10).ToVariable();
                Func<Params, Variable[]> f = args => [Loss.MeanSquaredError_simple(args.Get<Variable>("x"), x1)];
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x0)));
            }

            [Test]
            public void Test_Backward2()
            {
                var x0 = xp.random.rand(100).ToVariable();
                var x1 = xp.random.rand(100).ToVariable();
                Func<Params, Variable[]> f = args => [Loss.MeanSquaredError_simple(args.Get<Variable>("x"), x1)];
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x0)));
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
                var x0 = xp.array([0.0, 1.0, 2.0]).ToVariable();
                var x1 = xp.array([0.0, 1.0, 2.0]).ToVariable();
                var expected = ((x0 - x1) * (x0 - x1)).Data.Value.sum() / x0.size;
                var y = Loss.MeanSquaredError_simple(x0, x1);
                Assert.IsTrue(Utils.array_allclose(y.Data.Value, expected));
            }

            [Test]
            public void Test_Backward1()
            {
                var x0 = xp.random.rand(10).ToVariable();
                var x1 = xp.random.rand(10).ToVariable();
                Func<Params, Variable[]> f = args => [Loss.MeanSquaredError_simple(args.Get<Variable>("x"), x1)];
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x0)));
            }

            [Test]
            public void Test_Backward2()
            {
                var x0 = xp.random.rand(100).ToVariable();
                var x1 = xp.random.rand(100).ToVariable();
                Func<Params, Variable[]> f = args => [Loss.MeanSquaredError_simple(args.Get<Variable>("x"), x1)];
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x0)));
            }
        }
    }
}
