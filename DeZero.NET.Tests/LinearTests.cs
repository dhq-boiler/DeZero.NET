﻿using DeZero.NET.Core;
using DeZero.NET.Extensions;
using DeZero.NET.Functions;
using Python.Runtime;

namespace DeZero.NET.Tests
{
    public class LinearTests
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
                Variable b = null;
                var y = Linear.Invoke(x, w, b)[0];

                var res = y.Data.Value;
                var expected = xp.array([[14, 32], [32, 77]]);
                Assert.That(Utils.array_allclose(res, expected));
            }

            [Test]
            public void Test_Forward2()
            {
                var x = xp.array([[1, 2, 3], [4, 5, 6]]).astype("f").ToVariable();
                var W = x.Data.Value.T.ToVariable();
                Variable b = null;
                var y = Linear.Invoke(x, W, b)[0];

                var cy = Chainer.CF.linear(x.Data.Value, W.Data.Value.T);
                Assert.That(Utils.array_allclose(y.Data.Value, cy));
            }

            [Test]
            public void Test_Forward3()
            {
                var layer = new Chainer.Links.Linear(3, 2);
                var x = xp.array([[1, 2, 3], [4, 5, 6]]).astype("f").ToVariable();
                var W = layer.W.T.ToVariable();
                var b = layer.b.ToVariable();
                var y = Linear.Invoke(x, W, b)[0];

                var cy = layer.__call__(x.Data.Value);
                Assert.That(Utils.array_allclose(y.Data.Value, cy));
            }

            [Test]
            public void Test_Backward1()
            {
                var x = xp.random.randn(3, 2).ToVariable();
                var W = xp.random.randn(2, 3).ToVariable();
                var b = xp.random.randn(3).ToVariable();
                Func<Params, Variable[]> f = args => Linear.Invoke(args.Get<Variable>("x"), W, b);
                Assert.That(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x)));
            }

            [Test]
            public void Test_Backward2()
            {
                var x = xp.random.randn(100, 200).ToVariable();
                var W = xp.random.randn(200, 300).ToVariable();
                Variable b = null;
                Func<Params, Variable[]> f = args => Linear.Invoke(args.Get<Variable>("x"), W, b);
                Assert.That(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x)));
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
                Variable b = null;
                var y = Linear.Invoke(x, w, b)[0];

                var res = y.Data.Value;
                var expected = xp.array([[14, 32], [32, 77]]);
                Assert.That(Utils.array_allclose(res, expected));
            }

            [Test]
            public void Test_Forward2()
            {
                var x = xp.array([[1, 2, 3], [4, 5, 6]]).astype("f").ToVariable();
                var W = x.Data.Value.T.ToVariable();
                Variable b = null;
                var y = Linear.Invoke(x, W, b)[0];

                var cy = Chainer.CF.linear(x.Data.Value, W.Data.Value.T);
                Assert.That(Utils.array_allclose(y.Data.Value, cy));
            }

            [Test]
            public void Test_Forward3()
            {
                var layer = new Chainer.Links.Linear(3, 2);
                var x = xp.array([[1, 2, 3], [4, 5, 6]]).astype("f").ToVariable();
                var W = layer.W.T.ToVariable();
                var b = layer.b.ToVariable();
                var y = Linear.Invoke(x, W, b)[0];

                var cy = layer.__call__(x.Data.Value);
                Assert.That(Utils.array_allclose(y.Data.Value, cy));
            }

            [Test]
            public void Test_Backward1()
            {
                var x = xp.random.randn(3, 2).ToVariable();
                var W = xp.random.randn(2, 3).ToVariable();
                var b = xp.random.randn(3).ToVariable();
                Func<Params, Variable[]> f = args => Linear.Invoke(args.Get<Variable>("x"), W, b);
                Assert.That(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x)));
            }

            [Test]
            public void Test_Backward2()
            {
                var x = xp.random.randn(100, 200).ToVariable();
                var W = xp.random.randn(200, 300).ToVariable();
                Variable b = null;
                Func<Params, Variable[]> f = args => Linear.Invoke(args.Get<Variable>("x"), W, b);
                Assert.That(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x)));
            }
        }
    }
}
