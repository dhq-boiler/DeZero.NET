using DeZero.NET.Core;
using DeZero.NET.Extensions;
using DeZero.NET.Functions;
using Python.Runtime;

namespace DeZero.NET.Tests
{
    public class ReLuTests
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
                var x = xp.array([[-1, 0], [2, -3], [-2, 1]], xp.float32).ToVariable();
                var res = ReLU.Invoke(x)[0];
                var ans = xp.array([[0, 0], [2, 0], [0, 1]], xp.float32);
                Assert.That(Utils.array_allclose(res.Data.Value, ans));
            }

            [Test]
            public void Test_Backward1()
            {
                var x_data = xp.array([[-1, 1, 2], [-1, 2, 4]]).ToVariable();
                Assert.That(Utils.gradient_check(new ReLU(), Params.New.SetPositionalArgs(x_data)));
            }

            [Test]
            public void Test_Backward2()
            {
                xp.random.seed(0);
                var x_data = (xp.random.rand(10, 10) * 100).ToVariable();
                Assert.That(Utils.gradient_check(new ReLU(), Params.New.SetPositionalArgs(x_data)));
            }

            [Test]
            public void Test_Backward3()
            {
                xp.random.seed(0);
                var x_data = (xp.random.rand(10, 10, 10) * 100).ToVariable();
                Assert.That(Utils.gradient_check(new ReLU(), Params.New.SetPositionalArgs(x_data)));
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
                var x = xp.array([[-1, 0], [2, -3], [-2, 1]], xp.float32).ToVariable();
                var res = ReLU.Invoke(x)[0];
                var ans = xp.array([[0, 0], [2, 0], [0, 1]], xp.float32);
                Assert.That(Utils.array_allclose(res.Data.Value, ans));
            }

            [Test]
            public void Test_Backward1()
            {
                var x_data = xp.array([[-1, 1, 2], [-1, 2, 4]]).ToVariable();
                Assert.That(Utils.gradient_check(new ReLU(), Params.New.SetPositionalArgs(x_data)));
            }

            [Test]
            public void Test_Backward2()
            {
                xp.random.seed(0);
                var x_data = (xp.random.rand(10, 10) * 100).ToVariable();
                Assert.That(Utils.gradient_check(new ReLU(), Params.New.SetPositionalArgs(x_data)));
            }

            [Test]
            public void Test_Backward3()
            {
                xp.random.seed(0);
                var x_data = (xp.random.rand(10, 10, 10) * 100).ToVariable();
                Assert.That(Utils.gradient_check(new ReLU(), Params.New.SetPositionalArgs(x_data)));
            }
        }
    }
}
