using DeZero.NET.Core;
using DeZero.NET.Extensions;
using DeZero.NET.Functions;
using Python.Runtime;

namespace DeZero.NET.Tests
{
    public class TransposeTests
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
                var x = xp.array([[1, 2, 3], [4, 5, 6]]).ToVariable();
                var y = Transpose.Invoke(x)[0];
                Assert.That(y.Shape, Is.EqualTo(new Shape(3, 2)));
            }

            [Test]
            public void Test_Backward1()
            {
                var x = xp.array([[1, 2, 3], [4, 5, 6]]).ToVariable();
                Assert.That(Utils.gradient_check(new Transpose(), Params.New.SetPositionalArgs(x)));
            }

            [Test]
            public void Test_Backward2()
            {
                var x = xp.array([1, 2, 3]).ToVariable();
                Assert.That(Utils.gradient_check(new Transpose(), Params.New.SetPositionalArgs(x)));
            }

            [Test]
            public void Test_Backward3()
            {
                var x = xp.random.randn(10, 5).ToVariable();
                Assert.That(Utils.gradient_check(new Transpose(), Params.New.SetPositionalArgs(x)));
            }

            [Test]
            public void Test_Backward4()
            {
                var x = xp.array([1, 2]).ToVariable();
                Assert.That(Utils.gradient_check(new Transpose(), Params.New.SetPositionalArgs(x)));
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
                var x = xp.array([[1, 2, 3], [4, 5, 6]]).ToVariable();
                var y = Transpose.Invoke(x)[0];
                Assert.That(y.Shape, Is.EqualTo(new Shape(3, 2)));
            }

            [Test]
            public void Test_Backward1()
            {
                var x = xp.array([[1, 2, 3], [4, 5, 6]]).ToVariable();
                Assert.That(Utils.gradient_check(new Transpose(), Params.New.SetPositionalArgs(x)));
            }

            [Test]
            public void Test_Backward2()
            {
                var x = xp.array([1, 2, 3]).ToVariable();
                Assert.That(Utils.gradient_check(new Transpose(), Params.New.SetPositionalArgs(x)));
            }

            [Test]
            public void Test_Backward3()
            {
                var x = xp.random.randn(10, 5).ToVariable();
                Assert.That(Utils.gradient_check(new Transpose(), Params.New.SetPositionalArgs(x)));
            }

            [Test]
            public void Test_Backward4()
            {
                var x = xp.array([1, 2]).ToVariable();
                Assert.That(Utils.gradient_check(new Transpose(), Params.New.SetPositionalArgs(x)));
            }
        }
    }
}
