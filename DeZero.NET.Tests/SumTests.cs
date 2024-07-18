﻿using DeZero.NET.Core;
using DeZero.NET.Functions;
using Python.Runtime;

namespace DeZero.NET.Tests
{
    public class SumTests
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
            public void Test_DataType()
            {
                var x = xp.random.rand(10).ToVariable();
                var y = Sum.Invoke(x)[0];
                Assert.IsFalse(xp.isscalar(y.Data.Value));
            }

            [Test]
            public void Test_Forward1()
            {
                var x = xp.array(2.0).ToVariable();
                var y = Sum.Invoke(x)[0];
                var expected = xp.sum(x.Data.Value);
                Assert.IsTrue(Utils.array_allclose(y.Data.Value, expected));
            }

            [Test]
            public void Test_Forward2()
            {
                var x = xp.random.rand(10, 20, 30).ToVariable();
                var y = Sum.Invoke(x, axis: new Axis(1))[0];
                var expected = xp.sum(x.Data.Value, axis: 1);
                Assert.IsTrue(Utils.array_allclose(y.Data.Value, expected));
            }

            [Test]
            public void Test_Forward3()
            {
                var x = xp.random.rand(10, 20, 30).ToVariable();
                var y = Sum.Invoke(x, axis: new Axis(1), keepdims: true)[0];
                var expected = xp.sum(x.Data.Value, axis: 1, keepdims: true);
                Assert.IsTrue(Utils.array_allclose(y.Data.Value, expected));
            }

            [Test]
            public void Test_Backward1()
            {
                var x_data = xp.random.rand(10).ToVariable();
                Func<Params, Variable[]> f = args => Sum.Invoke(args.Get<Variable>("x"));
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x_data)));
            }

            [Test]
            public void Test_Backward2()
            {
                var x_data = xp.random.rand(10, 10).ToVariable();
                Func<Params, Variable[]> f = args => Sum.Invoke(args.Get<Variable>("x"), axis: new Axis(1));
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x_data)));
            }

            [Test]
            public void Test_Backward3()
            {
                var x_data = xp.random.rand(10, 20, 20).ToVariable();
                Func<Params, Variable[]> f = args => Sum.Invoke(args.Get<Variable>("x"), axis: new Axis(2));
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x_data)));
            }

            [Test]
            public void Test_Backward4()
            {
                var x_data = xp.random.rand(10, 20, 20).ToVariable();
                Func<Params, Variable[]> f = args => Sum.Invoke(args.Get<Variable>("x"), axis: null);
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x_data)));
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
            public void Test_DataType()
            {
                var x = xp.random.rand(10).ToVariable();
                var y = Sum.Invoke(x)[0];
                Assert.IsFalse(xp.isscalar(y.Data.Value));
            }

            [Test]
            public void Test_Forward1()
            {
                var x = xp.array(2.0).ToVariable();
                var y = Sum.Invoke(x)[0];
                var expected = xp.sum(x.Data.Value);
                Assert.IsTrue(Utils.array_allclose(y.Data.Value, expected));
            }

            [Test]
            public void Test_Forward2()
            {
                var x = xp.random.rand(10, 20, 30).ToVariable();
                var y = Sum.Invoke(x, axis: new Axis(1))[0];
                var expected = xp.sum(x.Data.Value, axis: 1);
                Assert.IsTrue(Utils.array_allclose(y.Data.Value, expected));
            }

            [Test]
            public void Test_Forward3()
            {
                var x = xp.random.rand(10, 20, 30).ToVariable();
                var y = Sum.Invoke(x, axis: new Axis(1), keepdims: true)[0];
                var expected = xp.sum(x.Data.Value, axis: 1, keepdims: true);
                Assert.IsTrue(Utils.array_allclose(y.Data.Value, expected));
            }

            [Test]
            public void Test_Backward1()
            {
                var x_data = xp.random.rand(10).ToVariable();
                Func<Params, Variable[]> f = args => Sum.Invoke(args.Get<Variable>("x"));
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x_data)));
            }

            [Test]
            public void Test_Backward2()
            {
                var x_data = xp.random.rand(10, 10).ToVariable();
                Func<Params, Variable[]> f = args => Sum.Invoke(args.Get<Variable>("x"), axis: new Axis(1));
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x_data)));
            }

            [Test]
            public void Test_Backward3()
            {
                var x_data = xp.random.rand(10, 20, 20).ToVariable();
                Func<Params, Variable[]> f = args => Sum.Invoke(args.Get<Variable>("x"), axis: new Axis(2));
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x_data)));
            }

            [Test]
            public void Test_Backward4()
            {
                var x_data = xp.random.rand(10, 20, 20).ToVariable();
                Func<Params, Variable[]> f = args => Sum.Invoke(args.Get<Variable>("x"), axis: null);
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x_data)));
            }
        }
    }
    public class SumToTests
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
                var x = xp.random.rand(10).ToVariable();
                var y = SumTo.Invoke(x, new Shape(1))[0];
                var expected = xp.sum(x.Data.Value);
                Assert.IsTrue(Utils.array_allclose(y.Data.Value, expected));
            }

            [Test]
            public void Test_Forward2()
            {
                var x = xp.array([[1f, 2f, 3f], [4f, 5f, 6f]]).ToVariable();
                var y = SumTo.Invoke(x, new Shape(1, 3))[0];
                var expected = xp.sum(x.Data.Value, axis: 0, keepdims: true);
                Assert.IsTrue(Utils.array_allclose(y.Data.Value, expected));
            }

            [Test]
            public void Test_Forward3()
            {
                var x = xp.random.rand(10).ToVariable();
                var y = SumTo.Invoke(x, new Shape(10))[0];
                var expected = x.Data.Value;
                Assert.IsTrue(Utils.array_allclose(y.Data.Value, expected));
            }

            [Test]
            public void Test_Backward1()
            {
                var x_data = xp.random.rand(10).ToVariable();
                Func<Params, Variable[]> f = args => SumTo.Invoke(args.Get<Variable>("x"), new Shape(1));
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x_data)));
            }

            [Test]
            public void Test_Backward2()
            {
                var x_data = (xp.random.rand(10, 10) * 10).ToVariable();
                Func<Params, Variable[]> f = args => SumTo.Invoke(args.Get<Variable>("x"), new Shape(10));
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x_data)));
            }

            [Test]
            public void Test_Backward3()
            {
                var x_data = (xp.random.rand(10, 20, 20) * 100).ToVariable();
                Func<Params, Variable[]> f = args => SumTo.Invoke(args.Get<Variable>("x"), new Shape(10));
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x_data)));
            }

            [Test]
            public void Test_Backward4()
            {
                var x_data = xp.random.rand(10).ToVariable();
                Func<Params, Variable[]> f = args => [SumTo.Invoke(args.Get<Variable>("x"), new Shape(10))[0] + 1];
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x_data)));
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
                var x = xp.random.rand(10).ToVariable();
                var y = SumTo.Invoke(x, new Shape(1))[0];
                var expected = xp.sum(x.Data.Value);
                Assert.IsTrue(Utils.array_allclose(y.Data.Value, expected));
            }

            [Test]
            public void Test_Forward2()
            {
                var x = xp.array([[1f, 2f, 3f], [4f, 5f, 6f]]).ToVariable();
                var y = SumTo.Invoke(x, new Shape(1, 3))[0];
                var expected = xp.sum(x.Data.Value, axis: 0, keepdims: true);
                Assert.IsTrue(Utils.array_allclose(y.Data.Value, expected));
            }

            [Test]
            public void Test_Forward3()
            {
                var x = xp.random.rand(10).ToVariable();
                var y = SumTo.Invoke(x, new Shape(10))[0];
                var expected = x.Data.Value;
                Assert.IsTrue(Utils.array_allclose(y.Data.Value, expected));
            }

            [Test]
            public void Test_Backward1()
            {
                var x_data = xp.random.rand(10).ToVariable();
                Func<Params, Variable[]> f = args => SumTo.Invoke(args.Get<Variable>("x"), new Shape(1));
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x_data)));
            }

            [Test]
            public void Test_Backward2()
            {
                var x_data = (xp.random.rand(10, 10) * 10).ToVariable();
                Func<Params, Variable[]> f = args => SumTo.Invoke(args.Get<Variable>("x"), new Shape(10));
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x_data)));
            }

            [Test]
            public void Test_Backward3()
            {
                var x_data = (xp.random.rand(10, 20, 20) * 100).ToVariable();
                Func<Params, Variable[]> f = args => SumTo.Invoke(args.Get<Variable>("x"), new Shape(10));
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x_data)));
            }

            [Test]
            public void Test_Backward4()
            {
                var x_data = xp.random.rand(10).ToVariable();
                Func<Params, Variable[]> f = args => [SumTo.Invoke(args.Get<Variable>("x"), new Shape(10))[0] + 1];
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x_data)));
            }
        }
    }
}
