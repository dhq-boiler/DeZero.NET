using DeZero.NET.Core;
using DeZero.NET.Functions;
using Python.Runtime;

namespace DeZero.NET.Tests
{
    public class MatMulTests
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
                var w = x.Data.Value.T.ToVariable();
                var y = MatMul.Invoke(x, w)[0];
                var res = y.Data.Value;
                var expected = xp.array([[14, 32], [32, 77]]);
                Assert.IsTrue(Utils.array_allclose(res, expected));
            }

            [Test]
            public void Test_Backward1()
            {
                var x = xp.random.randn(3, 2).ToVariable();
                var w = xp.random.randn(2, 3).ToVariable();
                Func<Params, Variable[]> f = args => [MatMul.Invoke(args.Get<Variable>("x"), w)[0]];
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x)));
            }

            [Test]
            public void Test_Backward2()
            {
                var x_data = xp.random.randn(10, 1).ToVariable();
                var w_data = xp.random.randn(1, 5).ToVariable();
                Func<Params, Variable[]> f = args => [MatMul.Invoke(x_data, args.Get<Variable>("x"))[0]];
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(w_data)));
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
                var w = x.Data.Value.T.ToVariable();
                var y = MatMul.Invoke(x, w)[0];
                var res = y.Data.Value;
                var expected = xp.array([[14, 32], [32, 77]]);
                Assert.IsTrue(Utils.array_allclose(res, expected));
            }

            [Test]
            public void Test_Backward1()
            {
                var x = xp.random.randn(3, 2).ToVariable();
                var w = xp.random.randn(2, 3).ToVariable();
                Func<Params, Variable[]> f = args => [MatMul.Invoke(args.Get<Variable>("x"), w)[0]];
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x)));
            }

            [Test]
            public void Test_Backward2()
            {
                var x_data = xp.random.randn(10, 1).ToVariable();
                var w_data = xp.random.randn(1, 5).ToVariable();
                Func<Params, Variable[]> f = args => [MatMul.Invoke(x_data, args.Get<Variable>("x"))[0]];
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(w_data)));
            }
        }
    }
}
