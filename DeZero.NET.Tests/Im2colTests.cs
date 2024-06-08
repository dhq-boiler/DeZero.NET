using DeZero.NET.Core;
using DeZero.NET.Functions;
using Python.Runtime;

namespace DeZero.NET.Tests
{
    public class Im2colTests
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
                int n = 1, c = 1, h = 3, w = 3;
                var x = xp.arange(n * c * h * w).reshape(n, c, h, w).ToVariable();
                var y = Im2col.Invoke(x, (3, 3), (3, 3), (0, 0), toMatrix: true);
                var expected = xp.array([[0, 1, 2, 3, 4, 5, 6, 7, 8]]);

                var res = Utils.array_equal(y.Data.Value, expected);
                Assert.IsTrue(res);
            }

            [Test]
            public void Test_Backward1()
            {
                int n = 1, c = 1, h = 3, w = 3;
                var x = xp.arange(n * c * h * w).reshape(n, c, h, w).ToVariable();
                Func<Params, Variable[]> f = args => [Im2col.Invoke(args.Get<Variable>("x"), (3, 3), (3, 3), (0, 0), toMatrix: true)];
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x)));
            }

            [Test]
            public void Test_Backward2()
            {
                int n = 1, c = 1, h = 3, w = 3;
                var x = xp.arange(n * c * h * w).reshape(n, c, h, w).ToVariable();
                Func<Params, Variable[]> f = args => [Im2col.Invoke(args.Get<Variable>("x"), (3, 3), (3, 3), (0, 0), toMatrix: false)];
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
                int n = 1, c = 1, h = 3, w = 3;
                var x = xp.arange(n * c * h * w).reshape(n, c, h, w).ToVariable();
                var y = Im2col.Invoke(x, (3, 3), (3, 3), (0, 0), toMatrix: true);
                var expected = xp.array([[0, 1, 2, 3, 4, 5, 6, 7, 8]]);

                var res = Utils.array_equal(y.Data.Value, expected);
                Assert.IsTrue(res);
            }

            [Test]
            public void Test_Backward1()
            {
                int n = 1, c = 1, h = 3, w = 3;
                var x = xp.arange(n * c * h * w).reshape(n, c, h, w).ToVariable();
                Func<Params, Variable[]> f = args => [Im2col.Invoke(args.Get<Variable>("x"), (3, 3), (3, 3), (0, 0), toMatrix: true)];
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x)));
            }

            [Test]
            public void Test_Backward2()
            {
                int n = 1, c = 1, h = 3, w = 3;
                var x = xp.arange(n * c * h * w).reshape(n, c, h, w).ToVariable();
                Func<Params, Variable[]> f = args => [Im2col.Invoke(args.Get<Variable>("x"), (3, 3), (3, 3), (0, 0), toMatrix: false)];
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x)));
            }
        }
    }
}
