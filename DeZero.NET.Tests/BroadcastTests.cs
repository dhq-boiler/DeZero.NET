using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeZero.NET.Tests
{
    public class BroadcastTests
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
            public void Test_Shape_Check()
            {
                var x = xp.random.randn(1, 10).ToVariable();
                var b = xp.random.randn(10).ToVariable();
                var y = x + b;
                var loss = Functions.Sum.Invoke(y);
                loss[0].Backward();
                Assert.AreEqual(b.Grad.Value.Shape, b.Shape);
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
            public void Test_Shape_Check()
            {
                var x = xp.random.randn(1, 10).ToVariable();
                var b = xp.random.randn(10).ToVariable();
                var y = x + b;
                var loss = Functions.Sum.Invoke(y);
                loss[0].Backward();
                Assert.AreEqual(b.Grad.Value.Shape, b.Shape);
            }
        }
    }
}
