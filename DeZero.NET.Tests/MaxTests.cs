using DeZero.NET.Core;
using DeZero.NET.Extensions;
using DeZero.NET.Functions;
using Python.Runtime;

namespace DeZero.NET.Tests
{
    public class MaxTests
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
                var y = Max.Invoke(x)[0];
                var expected = xp.max(x.Data.Value);
                Assert.That(Utils.array_allclose(y.Data.Value, expected));
            }

            [Test]
            public void Test_Forward2()
            {
                var shape = new Shape(10, 20, 30);
                int[] axis = [1];
                var x = xp.random.rand(shape.Dimensions).ToVariable();
                var y = Max.Invoke(x, axis: axis)[0];
                var expected = xp.max(x.Data.Value, axis: axis);
                Assert.That(Utils.array_allclose(y.Data.Value, expected));
            }

            [Test]
            public void Test_Forward3()
            {
                var shape = new Shape(10, 20, 30);
                int[] axis = [0, 1];
                var x = xp.random.rand(shape.Dimensions).ToVariable();
                var y = Max.Invoke(x, axis: axis)[0];
                var expected = xp.max(x.Data.Value, axis: axis);
                Assert.That(Utils.array_allclose(y.Data.Value, expected));
            }

            [Test]
            public void Test_Forward4()
            {
                var shape = new Shape(10, 20, 30);
                int[] axis = [0, 1];
                var x = xp.random.rand(shape.Dimensions).ToVariable();
                var y = Max.Invoke(x, axis: axis, keepdims: true)[0];
                var expected = xp.max(x.Data.Value, axis: axis, keepdims: true);
                Assert.That(Utils.array_allclose(y.Data.Value, expected));
            }

            [Test]
            public void Test_Backward1()
            {
                var x_data = xp.random.rand(10).ToVariable();
                Func<Params, Variable[]> f = args => Max.Invoke(args.Get<Variable>("x"));
                Assert.That(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x_data)));
            }

            [Test]
            public void Test_Backward2()
            {
                var x_data = (xp.random.rand(10, 10) * 100).ToVariable();
                Func<Params, Variable[]> f = args => Max.Invoke(args.Get<Variable>("x"), axis: [1]);
                Assert.That(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x_data)));
            }

            [Test]
            public void Test_Backward3()
            {
                var x_data = (xp.random.rand(10, 20, 30) * 100).ToVariable();
                Func<Params, Variable[]> f = args => Max.Invoke(args.Get<Variable>("x"), axis: [1, 2]);
                Assert.That(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x_data)));
            }

            [Test]
            public void Test_Backward4()
            {
                var x_data = (xp.random.rand(10, 20, 30) * 100).ToVariable();
                Func<Params, Variable[]> f = args => Max.Invoke(args.Get<Variable>("x"), axis: null);
                Assert.That(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x_data)));
            }

            [Test]
            public void Test_Backward5()
            {
                var x_data = (xp.random.rand(10, 20, 30) * 100).ToVariable();
                Func<Params, Variable[]> f = args => Max.Invoke(args.Get<Variable>("x"), axis: null, keepdims: true);
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
                var x = xp.random.rand(10).ToVariable();
                var y = Max.Invoke(x)[0];
                var expected = xp.max(x.Data.Value);
                Assert.That(Utils.array_allclose(y.Data.Value, expected));
            }

            [Test]
            public void Test_Forward2()
            {
                var shape = new Shape(10, 20, 30);
                int[] axis = [1];
                var x = xp.random.rand(shape.Dimensions).ToVariable();
                var y = Max.Invoke(x, axis: axis)[0];
                var expected = xp.max(x.Data.Value, axis: axis);
                Assert.That(Utils.array_allclose(y.Data.Value, expected));
            }

            [Test]
            public void Test_Forward3()
            {
                var shape = new Shape(10, 20, 30);
                int[] axis = [0, 1];
                var x = xp.random.rand(shape.Dimensions).ToVariable();
                var y = Max.Invoke(x, axis: axis)[0];
                var expected = xp.max(x.Data.Value, axis: axis);
                Assert.That(Utils.array_allclose(y.Data.Value, expected));
            }

            [Test]
            public void Test_Forward4()
            {
                var shape = new Shape(10, 20, 30);
                int[] axis = [0, 1];
                var x = xp.random.rand(shape.Dimensions).ToVariable();
                var y = Max.Invoke(x, axis: axis, keepdims: true)[0];
                var expected = xp.max(x.Data.Value, axis: axis, keepdims: true);
                Assert.That(Utils.array_allclose(y.Data.Value, expected));
            }

            [Test]
            public void Test_Backward1()
            {
                var x_data = xp.random.rand(10).ToVariable();
                Func<Params, Variable[]> f = args => Max.Invoke(args.Get<Variable>("x"));
                Assert.That(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x_data)));
            }

            [Test]
            public void Test_Backward2()
            {
                var x_data = (xp.random.rand(10, 10) * 100).ToVariable();
                Func<Params, Variable[]> f = args => Max.Invoke(args.Get<Variable>("x"), axis: [1]);
                Assert.That(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x_data)));
            }

            [Test]
            public void Test_Backward3()
            {
                var x_data = (xp.random.rand(10, 20, 30) * 100).ToVariable();
                Func<Params, Variable[]> f = args => Max.Invoke(args.Get<Variable>("x"), axis: [1, 2]);
                Assert.That(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x_data)));
            }

            [Test]
            public void Test_Backward4()
            {
                var x_data = (xp.random.rand(10, 20, 30) * 100).ToVariable();
                Func<Params, Variable[]> f = args => Max.Invoke(args.Get<Variable>("x"), axis: null);
                Assert.That(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x_data)));
            }

            [Test]
            public void Test_Backward5()
            {
                var x_data = (xp.random.rand(10, 20, 30) * 100).ToVariable();
                Func<Params, Variable[]> f = args => Max.Invoke(args.Get<Variable>("x"), axis: null, keepdims: true);
                Assert.That(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x_data)));
            }
        }
    }
}
