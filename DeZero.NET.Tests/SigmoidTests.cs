using DeZero.NET.Core;
using DeZero.NET.Functions;
using DeZero.NET.Tests.Chainer;
using Python.Runtime;

namespace DeZero.NET.Tests
{
    public class SigmoidTests
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
                var x = xp.array([[0, 1, 2], [0, 2, 4]], xp.float32).ToVariable();
                var y2 = CF.sigmoid(x.Data.Value);
                var y = Sigmoid.Invoke(x)[0];
                Assert.IsTrue(Utils.array_allclose(y.Data.Value, y2));
            }

            [Test]
            public void Test_Forward2()
            {
                var x = xp.random.randn(10, 10).astype(xp.float32).ToVariable();
                var y2 = CF.sigmoid(x.Data.Value);
                var y = Sigmoid.Invoke(x)[0];
                Assert.IsTrue(Utils.array_allclose(y.Data.Value, y2));
            }

            [Test]
            public void Test_Backward1()
            {
                var x_data = xp.array([[0, 1, 2], [0, 2, 4]]).ToVariable();
                Assert.IsTrue(Utils.gradient_check(new Sigmoid(), Params.New.SetPositionalArgs(x_data)));
            }

            [Test]
            public void Test_Backward2()
            {
                xp.random.seed(0);
                var x_data = xp.random.rand(10, 10).ToVariable();
                Assert.IsTrue(Utils.gradient_check(new Sigmoid(), Params.New.SetPositionalArgs(x_data)));
            }

            [Test]
            public void Test_Backward3()
            {
                xp.random.seed(0);
                var x_data = xp.random.rand(10, 10, 10).ToVariable();
                Assert.IsTrue(Utils.gradient_check(new Sigmoid(), Params.New.SetPositionalArgs(x_data)));
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
                var x = xp.array([[0, 1, 2], [0, 2, 4]], xp.float32).ToVariable();
                var y2 = CF.sigmoid(x.Data.Value);
                var y = Sigmoid.Invoke(x)[0];
                Assert.IsTrue(Utils.array_allclose(y.Data.Value, y2));
            }

            [Test]
            public void Test_Forward2()
            {
                var x = xp.random.randn(10, 10).astype(xp.float32).ToVariable();
                var y2 = CF.sigmoid(x.Data.Value);
                var y = Sigmoid.Invoke(x)[0];
                Assert.IsTrue(Utils.array_allclose(y.Data.Value, y2));
            }

            [Test]
            public void Test_Backward1()
            {
                var x_data = xp.array([[0, 1, 2], [0, 2, 4]]).ToVariable();
                Assert.IsTrue(Utils.gradient_check(new Sigmoid(), Params.New.SetPositionalArgs(x_data)));
            }

            [Test]
            public void Test_Backward2()
            {
                xp.random.seed(0);
                var x_data = xp.random.rand(10, 10).ToVariable();
                Assert.IsTrue(Utils.gradient_check(new Sigmoid(), Params.New.SetPositionalArgs(x_data)));
            }

            [Test]
            public void Test_Backward3()
            {
                xp.random.seed(0);
                var x_data = xp.random.rand(10, 10, 10).ToVariable();
                Assert.IsTrue(Utils.gradient_check(new Sigmoid(), Params.New.SetPositionalArgs(x_data)));
            }
        }
    }
}
