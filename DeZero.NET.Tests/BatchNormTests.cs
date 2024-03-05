using DeZero.NET.Functions;
using DeZero.NET.Tests.Chainer;
using NUnit.Framework.Internal.Commands;
using Python.Runtime;

namespace DeZero.NET.Tests
{
    public class FixedBatchNormTests
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

            private (NDarray, NDarray, NDarray, NDarray, NDarray) GetParams(int N, int C, int? H = null, int? W = null,
                string dtype = "f")
            {
                var _dtype = Extensions.dtype(dtype);
                NDarray x, gamma, beta, mean, var;
                if (H is not null)
                {
                    x = xp.random.randn(N, C, H.Value, W.Value).astype(_dtype);
                }
                else
                {
                    x = xp.random.randn(N, C).astype(_dtype);
                }

                gamma = xp.random.randn(C).astype(_dtype);
                beta = xp.random.randn(C).astype(_dtype);
                mean = xp.random.randn(C).astype(_dtype);
                var = xp.abs(xp.random.randn(C).astype(_dtype));
                return (x, gamma, beta, mean, var);
            }

            private (NDarray, NDarray, NDarray, NDarray, NDarray) GetParams(Dtype dtype, int N, int C, int? H = null, int? W = null)
            {
                return GetParams(N, C, H, W, dtype.CupyDtype.ToString());
            }

            [Test]
            public void Test_Type1()
            {
                int N = 8, C = 3;
                var (x, gamma, beta, mean, var) = GetParams(N, C);
                Variable[] y;
                using (DeZero.TestMode())
                {
                    y = BatchNorm.Invoke(x.ToVariable(), gamma.ToVariable(), beta.ToVariable(), mean.ToVariable(), var.ToVariable());
                }
                Assert.That(y[0].Data.dtype, Is.EqualTo(xp.float32));
            }

            [Test]
            public void Test_Forward1()
            {
                int N = 8, C = 1;
                var (x, gamma, beta, mean, var) = GetParams(N, C);
                var cy = CF.fixed_batch_normalization(x, gamma, beta, mean, var);
                cy = new NDarray(cy.data);
                Variable[] y;
                using (DeZero.TestMode())
                {
                    y = BatchNorm.Invoke(x.ToVariable(), gamma.ToVariable(), beta.ToVariable(), mean.ToVariable(), var.ToVariable());
                }
                Assert.That(Utils.array_allclose(y[0].Data, cy));
            }

            [Test]
            public void Test_Forward_extra1()
            {
                int N = 8, C = 1;
                var x = xp.array([[1.0f], [2.0f], [3.0f], [4.0f], [5.0f], [6.0f], [7.0f], [8.0f]]);
                var gamma = xp.array([2.0f]);
                var beta = xp.array([3.0f]);
                var mean = xp.array([4.0f]);
                var var = xp.array([5.0f]);
                var cy = CF.fixed_batch_normalization(x, gamma, beta, mean, var);
                cy = new NDarray(cy.data);
                Variable[] y;
                using (DeZero.TestMode())
                {
                    y = BatchNorm.Invoke(x.ToVariable(), gamma.ToVariable(), beta.ToVariable(), mean.ToVariable(), var.ToVariable());
                }
                Assert.That(Utils.array_allclose(y[0].Data, cy));
            }

            [Test]
            public void Test_Forward2()
            {
                int N = 1, C = 10;
                var (x, gamma, beta, mean, var) = GetParams(N, C);
                var cy = CF.fixed_batch_normalization(x, gamma, beta, mean, var);
                cy = new NDarray(cy.data);
                Variable[] y;
                using (DeZero.TestMode())
                {
                    y = BatchNorm.Invoke(x.ToVariable(), gamma.ToVariable(), beta.ToVariable(), mean.ToVariable(), var.ToVariable());
                }
                Assert.That(Utils.array_allclose(y[0].Data, cy));
            }

            [Test]
            public void Test_Forward3()
            {
                int N = 20, C = 10;
                var (x, gamma, beta, mean, var) = GetParams(N, C);
                var cy = CF.fixed_batch_normalization(x, gamma, beta, mean, var);
                cy = new NDarray(cy.data);
                Variable[] y;
                using (DeZero.TestMode())
                {
                    y = BatchNorm.Invoke(x.ToVariable(), gamma.ToVariable(), beta.ToVariable(), mean.ToVariable(), var.ToVariable());
                }
                Assert.That(Utils.array_allclose(y[0].Data, cy));
            }

            [Test]
            public void Test_Forward4()
            {
                int N = 20, C = 10, H = 5, W = 5;
                var (x, gamma, beta, mean, var) = GetParams(N, C, H, W);
                var cy = CF.fixed_batch_normalization(x, gamma, beta, mean, var);
                cy = new NDarray(cy.data);
                Variable[] y;
                using (DeZero.TestMode())
                {
                    y = BatchNorm.Invoke(x.ToVariable(), gamma.ToVariable(), beta.ToVariable(), mean.ToVariable(), var.ToVariable());
                }
                Assert.That(Utils.array_allclose(y[0].Data, cy));
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


            private (NDarray, NDarray, NDarray, NDarray, NDarray) GetParams(int N, int C, int? H = null, int? W = null,
                string dtype = "f")
            {
                var _dtype = Extensions.dtype(dtype);
                NDarray x, gamma, beta, mean, var;
                if (H is not null)
                {
                    x = xp.random.randn(N, C, H.Value, W.Value).astype(_dtype);
                }
                else
                {
                    x = xp.random.randn(N, C).astype(_dtype);
                }

                gamma = xp.random.randn(C).astype(_dtype);
                beta = xp.random.randn(C).astype(_dtype);
                mean = xp.random.randn(C).astype(_dtype);
                var = xp.abs(xp.random.randn(C).astype(_dtype));
                return (x, gamma, beta, mean, var);
            }

            private (NDarray, NDarray, NDarray, NDarray, NDarray) GetParams(Dtype dtype, int N, int C, int? H = null, int? W = null)
            {
                return GetParams(N, C, H, W, dtype.CupyDtype.ToString());
            }

            [Test]
            public void Test_Type1()
            {
                int N = 8, C = 3;
                var (x, gamma, beta, mean, var) = GetParams(N, C);
                Variable[] y;
                using (DeZero.TestMode())
                {
                    y = BatchNorm.Invoke(x.ToVariable(), gamma.ToVariable(), beta.ToVariable(), mean.ToVariable(), var.ToVariable());
                }
                Assert.That(y[0].Data.dtype, Is.EqualTo(xp.float32));
            }

            [Test]
            public void Test_Forward1()
            {
                int N = 8, C = 1;
                var (x, gamma, beta, mean, var) = GetParams(N, C);
                var cy = CF.fixed_batch_normalization(x, gamma, beta, mean, var);
                cy = new NDarray(cy.data);
                Variable[] y;
                using (DeZero.TestMode())
                {
                    y = BatchNorm.Invoke(x.ToVariable(), gamma.ToVariable(), beta.ToVariable(), mean.ToVariable(), var.ToVariable());
                }
                Assert.That(Utils.array_allclose(y[0].Data, cy));
            }

            [Test]
            public void Test_Forward2()
            {
                int N = 1, C = 10;
                var (x, gamma, beta, mean, var) = GetParams(N, C);
                var cy = CF.fixed_batch_normalization(x, gamma, beta, mean, var);
                cy = new NDarray(cy.data);
                Variable[] y;
                using (DeZero.TestMode())
                {
                    y = BatchNorm.Invoke(x.ToVariable(), gamma.ToVariable(), beta.ToVariable(), mean.ToVariable(), var.ToVariable());
                }
                Assert.That(Utils.array_allclose(y[0].Data, cy));
            }

            [Test]
            public void Test_Forward3()
            {
                int N = 20, C = 10;
                var (x, gamma, beta, mean, var) = GetParams(N, C);
                var cy = CF.fixed_batch_normalization(x, gamma, beta, mean, var);
                cy = new NDarray(cy.data);
                Variable[] y;
                using (DeZero.TestMode())
                {
                    y = BatchNorm.Invoke(x.ToVariable(), gamma.ToVariable(), beta.ToVariable(), mean.ToVariable(), var.ToVariable());
                }
                Assert.That(Utils.array_allclose(y[0].Data, cy));
            }

            [Test]
            public void Test_Forward4()
            {
                int N = 20, C = 10, H = 5, W = 5;
                var (x, gamma, beta, mean, var) = GetParams(N, C, H, W);
                var cy = CF.fixed_batch_normalization(x, gamma, beta, mean, var);
                cy = new NDarray(cy.data);
                Variable[] y;
                using (DeZero.TestMode())
                {
                    y = BatchNorm.Invoke(x.ToVariable(), gamma.ToVariable(), beta.ToVariable(), mean.ToVariable(), var.ToVariable());
                }
                Assert.That(Utils.array_allclose(y[0].Data, cy));
            }
        }
    }

    public class BachNormTests
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

            private (NDarray, NDarray, NDarray, NDarray, NDarray) GetParams(int N, int C, int? H = null, int? W = null,
                string dtype = "f")
            {
                var _dtype = Extensions.dtype(dtype);
                NDarray x, gamma, beta, mean, var;
                if (H is not null)
                {
                    x = xp.random.randn(N, C, H.Value, W.Value).astype(_dtype);
                }
                else
                {
                    x = xp.random.randn(N, C).astype(_dtype);
                }

                gamma = xp.random.randn(C).astype(_dtype);
                beta = xp.random.randn(C).astype(_dtype);
                mean = xp.random.randn(C).astype(_dtype);
                var = xp.abs(xp.random.randn(C).astype(_dtype));
                return (x, gamma, beta, mean, var);
            }

            private (NDarray, NDarray, NDarray, NDarray, NDarray) GetParams(Dtype dtype, int N, int C, int? H = null, int? W = null)
            {
                return GetParams(N, C, H, W, dtype.CupyDtype.ToString());
            }

            [Test]
            public void Test_Type1()
            {
                int N = 8, C = 3;
                var (x, gamma, beta, mean, var) = GetParams(N, C);
                var y = BatchNorm.Invoke(x.ToVariable(), gamma.ToVariable(), beta.ToVariable(), mean.ToVariable(), var.ToVariable());
                Assert.That(y[0].Data.dtype, Is.TypeOf(xp.float32.GetType()));
            }

            [Test]
            public void Test_Forward1()
            {
                int N = 8, C = 1;
                var (x, gamma, beta, mean, var) = GetParams(N, C);
                var cy = CF.batch_normalization(x, gamma, beta, running_mean: mean, running_var: var);
                cy = new NDarray(cy.data);
                var y = BatchNorm.Invoke(x.ToVariable(), gamma.ToVariable(), beta.ToVariable(), mean.ToVariable(), var.ToVariable());
                Assert.That(Utils.array_allclose(y[0].Data, cy));
            }

            [Test]
            public void Test_Forward2()
            {
                int N = 1, C = 10;
                var (x, gamma, beta, mean, var) = GetParams(N, C);
                var cy = CF.batch_normalization(x, gamma, beta, running_mean: mean, running_var: var);
                cy = new NDarray(cy.data);
                var y = BatchNorm.Invoke(x.ToVariable(), gamma.ToVariable(), beta.ToVariable(), mean.ToVariable(), var.ToVariable());
                Assert.That(Utils.array_allclose(y[0].Data, cy));
            }

            [Test]
            public void Test_Forward3()
            {
                int N = 20, C = 10;
                var (x, gamma, beta, mean, var) = GetParams(N, C);
                var cy = CF.batch_normalization(x, gamma, beta, running_mean: mean, running_var: var);
                cy = new NDarray(cy.data);
                var y = BatchNorm.Invoke(x.ToVariable(), gamma.ToVariable(), beta.ToVariable(), mean.ToVariable(), var.ToVariable());
                Assert.That(Utils.array_allclose(y[0].Data, cy));
            }

            [Test]
            public void Test_Forward4()
            {
                int N = 20, C = 10, H = 5, W = 5;
                var (x, gamma, beta, mean, var) = GetParams(N, C, H, W);
                var cy = CF.batch_normalization(x, gamma, beta, running_mean: mean, running_var: var);
                cy = new NDarray(cy.data);
                var y = BatchNorm.Invoke(x.ToVariable(), gamma.ToVariable(), beta.ToVariable(), mean.ToVariable(), var.ToVariable());
                Assert.That(Utils.array_allclose(y[0].Data, cy));
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


            private (NDarray, NDarray, NDarray, NDarray, NDarray) GetParams(int N, int C, int? H = null, int? W = null,
                string dtype = "f")
            {
                var _dtype = Extensions.dtype(dtype);
                NDarray x, gamma, beta, mean, var;
                if (H is not null)
                {
                    x = xp.random.randn(N, C, H.Value, W.Value).astype(_dtype);
                }
                else
                {
                    x = xp.random.randn(N, C).astype(_dtype);
                }

                gamma = xp.random.randn(C).astype(_dtype);
                beta = xp.random.randn(C).astype(_dtype);
                mean = xp.random.randn(C).astype(_dtype);
                var = xp.abs(xp.random.randn(C).astype(_dtype));
                return (x, gamma, beta, mean, var);
            }

            private (NDarray, NDarray, NDarray, NDarray, NDarray) GetParams(Dtype dtype, int N, int C, int? H = null, int? W = null)
            {
                return GetParams(N, C, H, W, dtype.CupyDtype.ToString());
            }

            [Test]
            public void Test_Type1()
            {
                int N = 8, C = 3;
                var (x, gamma, beta, mean, var) = GetParams(N, C);
                var y = BatchNorm.Invoke(x.ToVariable(), gamma.ToVariable(), beta.ToVariable(), mean.ToVariable(), var.ToVariable());
                Assert.That(y[0].Data.dtype, Is.TypeOf(xp.float32.GetType()));
            }

            [Test]
            public void Test_Forward1()
            {
                int N = 8, C = 1;
                var (x, gamma, beta, mean, var) = GetParams(N, C);
                var cy = CF.batch_normalization(x, gamma, beta, running_mean: mean, running_var: var);
                cy = new NDarray(cy.data);
                var y = BatchNorm.Invoke(x.ToVariable(), gamma.ToVariable(), beta.ToVariable(), mean.ToVariable(), var.ToVariable());
                Assert.That(Utils.array_allclose(y[0].Data, cy));
            }

            [Test]
            public void Test_Forward2()
            {
                int N = 1, C = 10;
                var (x, gamma, beta, mean, var) = GetParams(N, C);
                var cy = CF.batch_normalization(x, gamma, beta, running_mean: mean, running_var: var);
                cy = new NDarray(cy.data);
                var y = BatchNorm.Invoke(x.ToVariable(), gamma.ToVariable(), beta.ToVariable(), mean.ToVariable(), var.ToVariable());
                Assert.That(Utils.array_allclose(y[0].Data, cy));
            }

            [Test]
            public void Test_Forward3()
            {
                int N = 20, C = 10;
                var (x, gamma, beta, mean, var) = GetParams(N, C);
                var cy = CF.batch_normalization(x, gamma, beta, running_mean: mean, running_var: var);
                cy = new NDarray(cy.data);
                var y = BatchNorm.Invoke(x.ToVariable(), gamma.ToVariable(), beta.ToVariable(), mean.ToVariable(), var.ToVariable());
                Assert.That(Utils.array_allclose(y[0].Data, cy));
            }

            [Test]
            public void Test_Forward4()
            {
                int N = 20, C = 10, H = 5, W = 5;
                var (x, gamma, beta, mean, var) = GetParams(N, C, H, W);
                var cy = CF.batch_normalization(x, gamma, beta, running_mean: mean, running_var: var);
                cy = new NDarray(cy.data);
                var y = BatchNorm.Invoke(x.ToVariable(), gamma.ToVariable(), beta.ToVariable(), mean.ToVariable(), var.ToVariable());
                Assert.That(Utils.array_allclose(y[0].Data, cy));
            }
        }
    }
}
