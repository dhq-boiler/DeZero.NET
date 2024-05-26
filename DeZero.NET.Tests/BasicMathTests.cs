using DeZero.NET.Core;
using DeZero.NET.Functions;
using Python.Runtime;

namespace DeZero.NET.Tests
{
    public class AddTests
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
                var x0 = xp.array([1, 2, 3]);
                var x1 = new Variable(xp.array([1, 2, 3]));
                var y = x0 + x1;
                var res = y.Data.Value;
                var expected = xp.array([2, 4, 6]);
                Assert.That(res, Is.EqualTo(expected));
            }

            [Test]
            public void Test_DataType()
            {
                var x = new Variable(xp.array(2.0));
                var y = x.pow(2);
                Assert.That(xp.isscalar(y.Data.Value), Is.False);
            }

            [Test]
            public void Test_Backward0()
            {
                var x = xp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).ToVariable();
                var y = xp.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]]).ToVariable(useCupy: false);
                Func<Params, Variable[]> _f = x => [x.Get<Variable>("x") + y];
                Function f = new Add(_f);
                Assert.That(Utils.gradient_check(f, Params.New.SetPositionalArgs(x, y)), Is.True);
            }

            [Test]
            public void Test_Backward1()
            {
                var x = xp.random.randn(3, 3).ToVariable();
                var y = xp.random.randn(3, 3).ToVariable(useCupy: false);
                Func<Params, Variable[]> _f = x => [x.Get<Variable>("x") + y];
                Function f = new Add(_f);
                Assert.That(Utils.gradient_check(f, Params.New.SetPositionalArgs(x, y)), Is.True);
            }

            [Test]
            public void Test_Backward2()
            {
                var x = xp.random.randn(3, 3).ToVariable();
                var y = xp.random.randn(3, 1).ToVariable(useCupy: false);
                Func<Params, Variable[]> _f = x => [x.Get<Variable>("x") + y];
                Function f = new Add(_f);
                Assert.That(Utils.gradient_check(f, Params.New.SetPositionalArgs(x, y)), Is.True);
            }

            [Test]
            public void Test_Backward3()
            {
                var x = xp.random.randn(3, 3).ToVariable();
                var y = xp.random.randn(3, 1).ToVariable();
                Function f = new Add();
                Assert.That(Utils.gradient_check(f, Params.New.SetPositionalArgs(x, y)), Is.True);
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
                var x0 = xp.array([1, 2, 3]);
                var x1 = new Variable(xp.array([1, 2, 3]));
                var y = x0 + x1;
                var res = y.Data.Value;
                var expected = xp.array([2, 4, 6]);
                Assert.That(res, Is.EqualTo(expected));
            }

            [Test]
            public void Test_DataType()
            {
                var x = new Variable(xp.array(2.0));
                var y = x.pow(2);
                Assert.That(xp.isscalar(y.Data.Value), Is.False);
            }

            [Test]
            public void Test_Backward0()
            {
                var x = xp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).ToVariable();
                var y = xp.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]]).ToVariable(useCupy: false);
                Func<Params, Variable[]> _f = x => [x.Get<Variable>("x") + y];
                Function f = new Add(_f);
                Assert.That(Utils.gradient_check(f, Params.New.SetPositionalArgs(x, y)), Is.True);
            }

            [Test]
            public void Test_Backward1()
            {
                var x = xp.random.randn(3, 3).ToVariable();
                var y = xp.random.randn(3, 3).ToVariable(useCupy: false);
                Func<Params, Variable[]> _f = x => [x.Get<Variable>("x") + y];
                Function f = new Add(_f);
                Assert.That(Utils.gradient_check(f, Params.New.SetPositionalArgs(x, y)), Is.True);
            }

            [Test]
            public void Test_Backward2()
            {
                var x = xp.random.randn(3, 3).ToVariable();
                var y = xp.random.randn(3, 1).ToVariable(useCupy: false);
                Func<Params, Variable[]> _f = x => [x.Get<Variable>("x") + y];
                Function f = new Add(_f);
                Assert.That(Utils.gradient_check(f, Params.New.SetPositionalArgs(x, y)), Is.True);
            }

            [Test]
            public void Test_Backward3()
            {
                var x = xp.random.randn(3, 3).ToVariable();
                var y = xp.random.randn(3, 1).ToVariable();
                Function f = new Add();
                Assert.That(Utils.gradient_check(f, Params.New.SetPositionalArgs(x, y)), Is.True);
            }
        }
    }

    public class MulTests
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
                var x0 = xp.array([1, 2, 3]);
                var x1 = new Variable(xp.array([1, 2, 3]));
                var y = x0 * x1;
                var res = y.Data.Value;
                var expected = xp.array([1, 4, 9]);
                Assert.That(res, Is.EqualTo(expected));
            }

            [Test]
            public void Test_Backward1()
            {
                var x = xp.random.randn(3, 3).ToVariable();
                var y = xp.random.randn(3, 3).ToVariable(useCupy: false);
                Func<Params, Variable[]> _f = x => [(x.Get<Variable>("x") * y)];
                Function f = new Mul(_f);
                Assert.That(Utils.gradient_check(f, Params.New.SetPositionalArgs(x, y)), Is.True);
            }

            [Test]
            public void Test_Backward2()
            {
                var x = xp.random.randn(3, 3).ToVariable();
                var y = xp.random.randn(3, 1).ToVariable(useCupy: false);
                Func<Params, Variable[]> _f = x => [x.Get<Variable>("x") * y];
                Function f = new Mul(_f);
                Assert.That(Utils.gradient_check(f, Params.New.SetPositionalArgs(x, y)), Is.True);
            }

            [Test]
            public void Test_Backward3()
            {
                var x = xp.random.randn(3, 3).ToVariable(useCupy: false);
                var y = xp.random.randn(3, 1).ToVariable();
                Func<Params, Variable[]> _f = y => [x * y.Get<Variable>("y")];
                Function f = new Mul(_f);
                Assert.That(Utils.gradient_check(f, Params.New.SetPositionalArgs(x, y)), Is.True);
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
                var x0 = xp.array([1, 2, 3]);
                var x1 = new Variable(xp.array([1, 2, 3]));
                var y = x0 * x1;
                var res = y.Data.Value;
                var expected = xp.array([1, 4, 9]);
                Assert.That(res, Is.EqualTo(expected));
            }

            [Test]
            public void Test_Backward1()
            {
                var x = xp.random.randn(3, 3).ToVariable();
                var y = xp.random.randn(3, 3).ToVariable(useCupy: false);
                Func<Params, Variable[]> _f = x => [(x.Get<Variable>("x") * y)];
                Function f = new Mul(_f);
                Assert.That(Utils.gradient_check(f, Params.New.SetPositionalArgs(x, y)), Is.True);
            }

            [Test]
            public void Test_Backward2()
            {
                var x = xp.random.randn(3, 3).ToVariable();
                var y = xp.random.randn(3, 1).ToVariable(useCupy: false);
                Func<Params, Variable[]> _f = x => [x.Get<Variable>("x") * y];
                Function f = new Mul(_f);
                Assert.That(Utils.gradient_check(f, Params.New.SetPositionalArgs(x, y)), Is.True);
            }

            [Test]
            public void Test_Backward3()
            {
                var x = xp.random.randn(3, 3).ToVariable(useCupy: false);
                var y = xp.random.randn(3, 1).ToVariable();
                Func<Params, Variable[]> _f = y => [x * y.Get<Variable>("y")];
                Function f = new Mul(_f);
                Assert.That(Utils.gradient_check(f, Params.New.SetPositionalArgs(x, y)), Is.True);
            }
        }
    }

    public class DivTests
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
                var x0 = xp.array([1, 2, 3]);
                var x1 = new Variable(xp.array([1, 2, 3]));
                var y = x0 / x1;
                var res = y.Data.Value;
                var expected = xp.array([1, 1, 1]);
                Assert.That(res, Is.EqualTo(expected));
            }

            [Test]
            public void Test_Backward1()
            {
                var x = xp.random.randn(3, 3).ToVariable();
                var y = xp.random.randn(3, 3).ToVariable(autoSwitch: true, useCupy: false);
                Func<Params, Variable[]> _f = x => [x.Get<Variable>(0) / y];
                Function f = new Function(_f);
                Assert.That(Utils.gradient_check(f, Params.New.SetPositionalArgs(x, y)), Is.True);
            }

            [Test]
            public void Test_Backward2()
            {
                var x = xp.random.randn(3, 3).ToVariable();
                var y = xp.random.randn(3, 1).ToVariable(autoSwitch: true, useCupy: false);
                Func<Params, Variable[]> _f = x => [x.Get<Variable>(0) / y];
                Function f = new Function(_f);
                Assert.That(Utils.gradient_check(f, Params.New.SetPositionalArgs(x, y)), Is.True);
            }

            [Test]
            public void Test_Backward3()
            {
                var x = xp.random.randn(3, 3).ToVariable(autoSwitch: true, useCupy: false);
                var y = xp.random.randn(3, 1).ToVariable();
                var args = Params.New.SetPositionalArgs(x).SetKeywordArg(y);
                Assert.That(Utils.gradient_check(new Div(), args), Is.True);
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
                var x0 = xp.array([1, 2, 3]);
                var x1 = new Variable(xp.array([1, 2, 3]));
                var y = x0 / x1;
                var res = y.Data.Value;
                var expected = xp.array([1, 1, 1]);
                Assert.That(res, Is.EqualTo(expected));
            }

            [Test]
            public void Test_Backward1()
            {
                var x = xp.random.randn(3, 3).ToVariable();
                var y = xp.random.randn(3, 3).ToVariable(autoSwitch: true, useCupy: false);
                Func<Params, Variable[]> _f = x => [x.Get<Variable>(0) / y];
                Function f = new Div(_f);
                Assert.That(Utils.gradient_check(f, Params.New.SetPositionalArgs(x, y)), Is.True);
            }

            [Test]
            public void Test_Backward2()
            {
                var x = xp.random.randn(3, 3).ToVariable();
                var y = xp.random.randn(3, 1).ToVariable(autoSwitch: true, useCupy: false);
                Func<Params, Variable[]> _f = x => [x.Get<Variable>(0) / y];
                Function f = new Div(_f);
                Assert.That(Utils.gradient_check(f, Params.New.SetPositionalArgs(x, y)), Is.True);
            }

            [Test]
            public void Test_Backward3()
            {
                var x = xp.random.randn(3, 3).ToVariable(autoSwitch: true, useCupy: false);
                var y = xp.random.randn(3, 1).ToVariable();
                Function f = new Div();
                Assert.That(Utils.gradient_check(f, Params.New.SetPositionalArgs(x).SetKeywordArg(y)), Is.True);
            }
        }
    }
}