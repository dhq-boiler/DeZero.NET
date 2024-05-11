using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DeZero.NET.Core;
using DeZero.NET.Functions;

namespace DeZero.NET.Tests
{
    public class GetItemTests
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
                var x_data = xp.arange(12).reshape(2, 2, 3);
                var x = x_data.ToVariable();
                var y = GetItem.Invoke(x, 0);
                Assert.IsTrue(Utils.array_allclose(y[0].Data, x_data[0]));
            }

            [Test]
            public void Test_Forward1a()
            {
                var x_data = xp.arange(12).reshape(2, 2, 3);
                var x = x_data.ToVariable();
                Variable[] y = [x.Data[0].ToVariable()];
                Assert.IsTrue(Utils.array_allclose(y[0].Data, x_data[0]));
            }

            [Test]
            public void Test_Forward2()
            {
                var x_data = xp.arange(12).reshape(2, 2, 3);
                var x = x_data.ToVariable();
                var y = GetItem.Invoke(x, 0, 0, new Slice(0, 2, 1));
                Assert.IsTrue(Utils.array_allclose(y[0].Data, x_data[0, 0, new Slice(0, 2, 1)]));
            }

            [Test]
            public void Test_Forward3()
            {
                var x_data = xp.arange(12).reshape(2, 2, 3);
                var x = x_data.ToVariable();
                var y = GetItem.Invoke(x, new Slice(0, 2)); //本当はnew Slice(..., 2)にしたかった
                Assert.IsTrue(Utils.array_allclose(y[0].Data, x_data[new Slice(0, 2)])); //本当はx_data[..., 2]にしたかった
            }

            [Test]
            public void Test_Backward1()
            {
                var x_data = xp.array([[1, 2, 3], [4, 5, 6]]);
                var slices = 1;
                Func<Params, Variable[]> f = args => GetItem.Invoke(args.Get<Variable>("x"), slices);
                Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x_data));
            }

            [Test]
            public void Test_Backward2()
            {
                var x_data = xp.arange(12).reshape(4, 3);
                var slices = new Slice(1, 3);
                Func<Params, Variable[]> f = args => GetItem.Invoke(args.Get<Variable>("x"), slices);
                Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x_data));
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
                var x_data = xp.arange(12).reshape(2, 2, 3);
                var x = x_data.ToVariable();
                var y = GetItem.Invoke(x, 0);
                Assert.IsTrue(Utils.array_allclose(y[0].Data, x_data[0]));
            }

            [Test]
            public void Test_Forward1a()
            {
                var x_data = xp.arange(12).reshape(2, 2, 3);
                var x = x_data.ToVariable();
                Variable[] y = [x.Data[0].ToVariable()];
                Assert.IsTrue(Utils.array_allclose(y[0].Data, x_data[0]));
            }

            [Test]
            public void Test_Forward2()
            {
                var x_data = xp.arange(12).reshape(2, 2, 3);
                var x = x_data.ToVariable();
                var y = GetItem.Invoke(x, 0, 0, new Slice(0, 2, 1));
                Assert.IsTrue(Utils.array_allclose(y[0].Data, x_data[0, 0, new Slice(0, 2, 1)]));
            }

            [Test]
            public void Test_Forward3()
            {
                var x_data = xp.arange(12).reshape(2, 2, 3);
                var x = x_data.ToVariable();
                var y = GetItem.Invoke(x, new Slice(0, 2)); //本当はnew Slice(..., 2)にしたかった
                Assert.IsTrue(Utils.array_allclose(y[0].Data, x_data[new Slice(0, 2)])); //本当はx_data[..., 2]にしたかった
            }

            [Test]
            public void Test_Backward1()
            {
                var x_data = xp.array([[1, 2, 3], [4, 5, 6]]);
                var slices = 1;
                Func<Params, Variable[]> f = args => GetItem.Invoke(args.Get<Variable>("x"), slices);
                Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x_data));
            }

            [Test]
            public void Test_Backward2()
            {
                var x_data = xp.arange(12).reshape(4, 3);
                var slices = new Slice(1, 3);
                Func<Params, Variable[]> f = args => GetItem.Invoke(args.Get<Variable>("x"), slices);
                Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x_data));
            }
        }
    }
}
