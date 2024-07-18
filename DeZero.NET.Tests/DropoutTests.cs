using DeZero.NET.Core;
using Python.Runtime;

namespace DeZero.NET.Tests
{
    public class DropoutTests
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
                var x = xp.random.randn(100, 100);
                var y = Utils.dropout(x.ToVariable(), dropout_ratio: 0.0);
                var res = Utils.array_equal(y.Data.Value, x);
                Assert.IsTrue(res);
            }

            [Test]
            public void Test_Forward2()
            {
                var x = xp.random.randn(100, 100).ToVariable();
                Variable y;
                using (Preferences.TestMode())
                {
                    y = Utils.dropout(x);
                }
                var res = Utils.array_equal(y.Data.Value, x.Data.Value);
                Assert.IsTrue(res);
            }

            [Test]
            public void Test_Backward1()
            {
                var x_data = xp.random.randn(10, 10).ToVariable();

                Variable f(Variable x)
                {
                    Numpy.np.random.seed(0);
                    return Utils.dropout(x, dropout_ratio: 0.5);
                }

                Assert.IsTrue(Utils.gradient_check(new Function(args => [f(args.Get<Variable>("x"))]), Params.New.SetPositionalArgs(x_data)));
            }

            [Test]
            public void Test_Backward2()
            {
                var x_data = xp.random.randn(10, 20).ToVariable();

                Variable f(Variable x)
                {
                    Numpy.np.random.seed(0);
                    return Utils.dropout(x, dropout_ratio: 0.99);
                }

                Assert.IsTrue(Utils.gradient_check(new Function(args => [f(args.Get<Variable>("x"))]), Params.New.SetPositionalArgs(x_data)));
            }

            [Test]
            public void Test_Backward3()
            {
                var x_data = xp.random.randn(10, 10).ToVariable();

                Variable f(Variable x)
                {
                    Numpy.np.random.seed(0);
                    return Utils.dropout(x, dropout_ratio: 0.0);
                }

                Assert.IsTrue(Utils.gradient_check(new Function(args => [f(args.Get<Variable>("x"))]), Params.New.SetPositionalArgs(x_data)));
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
                var x = xp.random.randn(100, 100);
                var y = Utils.dropout(x.ToVariable(), dropout_ratio: 0.0);
                var res = Utils.array_equal(y.Data.Value, x);
                Assert.IsTrue(res);
            }

            [Test]
            public void Test_Forward2()
            {
                var x = xp.random.randn(100, 100).ToVariable();
                Variable y;
                using (Preferences.TestMode())
                {
                    y = Utils.dropout(x);
                }
                var res = Utils.array_equal(y.Data.Value, x.Data.Value);
                Assert.IsTrue(res);
            }

            [Test]
            public void Test_Backward1()
            {
                var x_data = xp.random.randn(10, 10).ToVariable();

                Variable f(Variable x)
                {
                    Numpy.np.random.seed(0);
                    return Utils.dropout(x, dropout_ratio: 0.5);
                }

                Assert.IsTrue(Utils.gradient_check(new Function(args => [f(args.Get<Variable>("x"))]), Params.New.SetPositionalArgs(x_data)));
            }

            [Test]
            public void Test_Backward2()
            {
                var x_data = xp.random.randn(10, 20).ToVariable();

                Variable f(Variable x)
                {
                    Numpy.np.random.seed(0);
                    return Utils.dropout(x, dropout_ratio: 0.99);
                }

                Assert.IsTrue(Utils.gradient_check(new Function(args => [f(args.Get<Variable>("x"))]), Params.New.SetPositionalArgs(x_data)));
            }

            [Test]
            public void Test_Backward3()
            {
                var x_data = xp.random.randn(10, 10).ToVariable();

                Variable f(Variable x)
                {
                    Numpy.np.random.seed(0);
                    return Utils.dropout(x, dropout_ratio: 0.0);
                }

                Assert.IsTrue(Utils.gradient_check(new Function(args => [f(args.Get<Variable>("x"))]), Params.New.SetPositionalArgs(x_data)));
            }
        }
    }
}
