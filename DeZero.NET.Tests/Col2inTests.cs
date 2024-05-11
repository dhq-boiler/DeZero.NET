﻿using DeZero.NET.Core;
using DeZero.NET.Functions;
using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeZero.NET.Tests
{
    public class Col2inTests
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
            public void Test_Backward1()
            {
                int n = 1, c = 1, h = 3, w = 3;
                var x = xp.random.rand(1, 9).ToVariable();
                Func<Params, Variable[]> f = args => [Col2im.Invoke(args.Get<Variable>("x"), new Shape(n, c, h, w), (3, 3), (3, 3), (0, 0), toMatrix: true)];
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x)));
            }

            [Test]
            public void Test_Backward2()
            {
                int n = 1, c = 1, h = 3, w = 3;
                var x = xp.random.rand(1, 1, 3, 3, 1, 1).ToVariable();
                Func<Params, Variable[]> f = args => [Col2im.Invoke(args.Get<Variable>("x"), new Shape(n, c, h, w), (3, 3), (3, 3), (0, 0), toMatrix: false)];
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x)));
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
            public void Test_Backward1()
            {
                int n = 1, c = 1, h = 3, w = 3;
                var x = xp.random.rand(1, 9).ToVariable();
                Func<Params, Variable[]> f = args => [Col2im.Invoke(args.Get<Variable>("x"), new Shape(n, c, h, w), (3, 3), (3, 3), (0, 0), toMatrix: true)];
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x)));
            }

            [Test]
            public void Test_Backward2()
            {
                int n = 1, c = 1, h = 3, w = 3;
                var x = xp.random.rand(1, 1, 3, 3, 1, 1).ToVariable();
                Func<Params, Variable[]> f = args => [Col2im.Invoke(args.Get<Variable>("x"), new Shape(n, c, h, w), (3, 3), (3, 3), (0, 0), toMatrix: false)];
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetPositionalArgs(x)));
            }
        }
    }
}
