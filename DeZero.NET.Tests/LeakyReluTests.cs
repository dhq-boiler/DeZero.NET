﻿using DeZero.NET.Core;
using DeZero.NET.Functions;
using DeZero.NET.Tests.Chainer;
using Python.Runtime;

namespace DeZero.NET.Tests
{
    public class LeakyReluTests
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
                var res = LeakyRelu.Invoke(x)[0];
                var ans = xp.array([[-0.2, 0], [2, -0.6], [-0.4, 1]], xp.float32);
                Assert.IsTrue(Utils.array_allclose(res.Data, ans));
            }

            [Test]
            public void Test_Forward2()
            {
                xp.random.seed(0);
                var slope = 0.002;
                var x = xp.random.randn(100).ToVariable();
                var y2 = CF.leaky_relu(x.Data, slope);
                var y = LeakyRelu.Invoke(x, slope)[0];
                Assert.IsTrue(Utils.array_allclose(y.Data, y2));
            }

            [Test]
            public void Test_Backward1()
            {
                var x_data = xp.array([[-1, 1, 2], [-1, 2, 4]]).ToVariable();
                Assert.IsTrue(Utils.gradient_check(new LeakyRelu(), Params.New.SetPositionalArgs(x_data)));
            }

            [Test]
            public void Test_Backward2()
            {
                xp.random.seed(0);
                var x_data = (xp.random.rand(10, 10) * 100).ToVariable();
                Assert.IsTrue(Utils.gradient_check(new LeakyRelu(), Params.New.SetPositionalArgs(x_data)));
            }

            [Test]
            public void Test_Backward3()
            {
                xp.random.seed(0);
                var x_data = (xp.random.rand(10, 10, 10) * 100).ToVariable();
                Assert.IsTrue(Utils.gradient_check(new LeakyRelu(), Params.New.SetPositionalArgs(x_data)));
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
                var res = LeakyRelu.Invoke(x)[0];
                var ans = xp.array([[-0.2, 0], [2, -0.6], [-0.4, 1]], xp.float32);
                Assert.IsTrue(Utils.array_allclose(res.Data, ans));
            }

            [Test]
            public void Test_Forward2()
            {
                xp.random.seed(0);
                var slope = 0.002;
                var x = xp.random.randn(100).ToVariable();
                var y2 = CF.leaky_relu(x.Data, slope);
                var y = LeakyRelu.Invoke(x, slope)[0];
                Assert.IsTrue(Utils.array_allclose(y.Data, y2));
            }

            [Test]
            public void Test_Backward1()
            {
                var x_data = xp.array([[-1, 1, 2], [-1, 2, 4]]).ToVariable();
                Assert.IsTrue(Utils.gradient_check(new LeakyRelu(), Params.New.SetPositionalArgs(x_data)));
            }

            [Test]
            public void Test_Backward2()
            {
                xp.random.seed(0);
                var x_data = (xp.random.rand(10, 10) * 100).ToVariable();
                Assert.IsTrue(Utils.gradient_check(new LeakyRelu(), Params.New.SetPositionalArgs(x_data)));
            }

            [Test]
            public void Test_Backward3()
            {
                xp.random.seed(0);
                var x_data = (xp.random.rand(10, 10, 10) * 100).ToVariable();
                Assert.IsTrue(Utils.gradient_check(new LeakyRelu(), Params.New.SetPositionalArgs(x_data)));
            }
        }
    }
}