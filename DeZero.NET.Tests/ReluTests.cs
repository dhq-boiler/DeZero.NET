using DeZero.NET.Core;
using DeZero.NET.Functions;
using Python.Runtime;

namespace DeZero.NET.Tests
{
    public class ReluTests
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
                var x = xp.array([[-1, 0], [2, -3], [-2, 1]], xp.float32).ToVariable();
                var res = Relu.Invoke(x)[0];
                var ans = xp.array([[0, 0], [2, 0], [0, 1]], xp.float32);
                Assert.IsTrue(Utils.array_allclose(res.Data, ans));
            }

            [Test]
            public void Test_Backward1()
            {
                var x_data = xp.array([[-1, 1, 2], [-1, 2, 4]]).ToVariable();
                Assert.IsTrue(Utils.gradient_check(new Relu(), Params.New.SetPositionalArgs(x_data)));
            }

            [Test]
            public void Test_Backward2()
            {
                xp.random.seed(0);
                var x_data = (xp.random.rand(10, 10) * 100).ToVariable();
                Assert.IsTrue(Utils.gradient_check(new Relu(), Params.New.SetPositionalArgs(x_data)));
            }

            [Test]
            public void Test_Backward3()
            {
                xp.random.seed(0);
                var x_data = (xp.random.rand(10, 10, 10) * 100).ToVariable();
                Assert.IsTrue(Utils.gradient_check(new Relu(), Params.New.SetPositionalArgs(x_data)));
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
                var x = xp.array([[-1, 0], [2, -3], [-2, 1]], xp.float32).ToVariable();
                var res = Relu.Invoke(x)[0];
                var ans = xp.array([[0, 0], [2, 0], [0, 1]], xp.float32);
                Assert.IsTrue(Utils.array_allclose(res.Data, ans));
            }

            [Test]
            public void Test_Backward1()
            {
                var x_data = xp.array([[-1, 1, 2], [-1, 2, 4]]).ToVariable();
                Assert.IsTrue(Utils.gradient_check(new Relu(), Params.New.SetPositionalArgs(x_data)));
            }

            [Test]
            public void Test_Backward2()
            {
                xp.random.seed(0);
                var x_data = (xp.random.rand(10, 10) * 100).ToVariable();
                Assert.IsTrue(Utils.gradient_check(new Relu(), Params.New.SetPositionalArgs(x_data)));
            }

            [Test]
            public void Test_Backward3()
            {
                xp.random.seed(0);
                var x_data = (xp.random.rand(10, 10, 10) * 100).ToVariable();
                Assert.IsTrue(Utils.gradient_check(new Relu(), Params.New.SetPositionalArgs(x_data)));
            }
        }
    }
}
