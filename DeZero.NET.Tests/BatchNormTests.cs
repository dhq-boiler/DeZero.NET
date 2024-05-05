using System.Security.Cryptography.X509Certificates;
using DeZero.NET.Core;
using DeZero.NET.Functions;
using DeZero.NET.Tests.Chainer;
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

            private (Variable, Variable, Variable, Variable, Variable) GetParams(int N, int C, int? H = null, int? W = null,
                string dtype = "f")
            {
                var _dtype = Extensions.dtype(dtype);
                NDarray x, gamma, beta, mean, @var;
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
                @var = xp.abs(xp.random.randn(C).astype(_dtype));

                return (x.ToVariable(), gamma.ToVariable(), beta.ToVariable(), mean.ToVariable(), @var.ToVariable());
            }

            private (Variable, Variable, Variable, Variable, Variable) GetParams(Dtype dtype, int N, int C, int? H = null, int? W = null)
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
                    y = BatchNorm.Invoke(x, gamma, beta, mean, var).Item1;
                }
                Assert.That(y[0].Data.dtype, Is.EqualTo(xp.float32));
            }

            [Test]
            public void Test_Forward1()
            {
                int N = 8, C = 1;
                var (x, gamma, beta, mean, var) = GetParams(N, C);
                var cy = CF.fixed_batch_normalization(x.Data, gamma.Data, beta.Data, mean.Data, var.Data);
                Variable[] y;
                using (DeZero.TestMode())
                {
                    y = BatchNorm.Invoke(x, gamma, beta, mean, var).Item1;
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
                Variable[] y;
                using (DeZero.TestMode())
                {
                    y = BatchNorm.Invoke(x.ToVariable(), gamma.ToVariable(), beta.ToVariable(), mean.ToVariable(), var.ToVariable()).Item1;
                }
                Assert.That(Utils.array_allclose(y[0].Data, cy));
            }

            [Test]
            public void Test_Forward2()
            {
                int N = 1, C = 10;
                var (x, gamma, beta, mean, var) = GetParams(N, C);
                var cy = CF.fixed_batch_normalization(x.Data, gamma.Data, beta.Data, mean.Data, var.Data);
                Variable[] y;
                using (DeZero.TestMode())
                {
                    y = BatchNorm.Invoke(x, gamma, beta, mean, var).Item1;
                }
                Assert.That(Utils.array_allclose(y[0].Data, cy));
            }

            [Test]
            public void Test_Forward3()
            {
                int N = 20, C = 10;
                var (x, gamma, beta, mean, var) = GetParams(N, C);
                var cy = CF.fixed_batch_normalization(x.Data, gamma.Data, beta.Data, mean.Data, var.Data);
                Variable[] y;
                using (DeZero.TestMode())
                {
                    y = BatchNorm.Invoke(x, gamma, beta, mean, var).Item1;
                }
                Assert.That(Utils.array_allclose(y[0].Data, cy));
            }

            [Test]
            public void Test_Forward4()
            {
                int N = 20, C = 10, H = 5, W = 5;
                var (x, gamma, beta, mean, var) = GetParams(N, C, H, W);
                var cy = CF.fixed_batch_normalization(x.Data, gamma.Data, beta.Data, mean.Data, var.Data);
                Variable[] y;
                using (DeZero.TestMode())
                {
                    y = BatchNorm.Invoke(x, gamma, beta, mean, var).Item1;
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


            private (Variable, Variable, Variable, Variable, Variable) GetParams(int N, int C, int? H = null, int? W = null,
                string dtype = "f")
            {
                var _dtype = Extensions.dtype(dtype);
                NDarray x, gamma, beta, mean, @var;
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
                @var = xp.abs(xp.random.randn(C).astype(_dtype));

                return (x.ToVariable(), gamma.ToVariable(), beta.ToVariable(), mean.ToVariable(), @var.ToVariable());
            }

            private (Variable, Variable, Variable, Variable, Variable) GetParams(Dtype dtype, int N, int C, int? H = null, int? W = null)
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
                    y = BatchNorm.Invoke(x, gamma, beta, mean, var).Item1;
                }
                Assert.That(y[0].Data.dtype, Is.EqualTo(xp.float32));
            }

            [Test]
            public void Test_Forward1()
            {
                int N = 8, C = 1;
                var (x, gamma, beta, mean, var) = GetParams(N, C);
                var cy = CF.fixed_batch_normalization(x.Data, gamma.Data, beta.Data, mean.Data, var.Data);
                Variable[] y;
                using (DeZero.TestMode())
                {
                    y = BatchNorm.Invoke(x, gamma, beta, mean, var).Item1;
                }
                Assert.That(Utils.array_allclose(y[0].Data, cy));
            }

            [Test]
            public void Test_Forward2()
            {
                int N = 1, C = 10;
                var (x, gamma, beta, mean, var) = GetParams(N, C);
                var cy = CF.fixed_batch_normalization(x.Data, gamma.Data, beta.Data, mean.Data, var.Data);
                Variable[] y;
                using (DeZero.TestMode())
                {
                    y = BatchNorm.Invoke(x, gamma, beta, mean, var).Item1;
                }
                Assert.That(Utils.array_allclose(y[0].Data, cy));
            }

            [Test]
            public void Test_Forward3()
            {
                int N = 20, C = 10;
                var (x, gamma, beta, mean, var) = GetParams(N, C);
                var cy = CF.fixed_batch_normalization(x.Data, gamma.Data, beta.Data, mean.Data, var.Data);
                Variable[] y;
                using (DeZero.TestMode())
                {
                    y = BatchNorm.Invoke(x, gamma, beta, mean, var).Item1;
                }
                Assert.That(Utils.array_allclose(y[0].Data, cy));
            }

            [Test]
            public void Test_Forward4()
            {
                int N = 20, C = 10, H = 5, W = 5;
                var (x, gamma, beta, mean, var) = GetParams(N, C, H, W);
                var cy = CF.fixed_batch_normalization(x.Data, gamma.Data, beta.Data, mean.Data, var.Data);
                Variable[] y;
                using (DeZero.TestMode())
                {
                    y = BatchNorm.Invoke(x, gamma, beta, mean, var).Item1;
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

            private (Variable, Variable, Variable, Variable, Variable) GetParams(int N, int C, int? H = null, int? W = null,
                string dtype = "f")
            {
                var _dtype = Extensions.dtype(dtype);
                NDarray x, gamma, beta, mean, @var;
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
                @var = xp.abs(xp.random.randn(C).astype(_dtype));

                return (x.ToVariable(), gamma.ToVariable(), beta.ToVariable(), mean.ToVariable(), @var.ToVariable());
            }

            private (Variable, Variable, Variable, Variable, Variable) GetParams(Dtype dtype, int N, int C, int? H = null, int? W = null)
            {
                return GetParams(N, C, H, W, dtype.CupyDtype.ToString());
            }

            private (Variable, Variable, Variable, Variable, Variable) GetOnesParams(int N, int C, int? H = null, int? W = null,
                string dtype = "f")
            {
                var _dtype = Extensions.dtype(dtype);
                NDarray x, gamma, beta, mean, @var;
                if (H is not null)
                {
                    x = xp.ones(new Shape(N, C, H.Value, W.Value)).astype(_dtype);
                }
                else
                {
                    x = xp.ones(new Shape(N, C)).astype(_dtype);
                }

                gamma = (xp.ones(C) * 2).astype(_dtype);
                beta = xp.ones(C).astype(_dtype);
                mean = xp.ones(C).astype(_dtype);
                @var = xp.ones(C).astype(_dtype);

                return (x.ToVariable(), gamma.ToVariable(), beta.ToVariable(), mean.ToVariable(), @var.ToVariable());
            }

            private (Variable, Variable, Variable, Variable, Variable) GetOnesParams(Dtype dtype, int N, int C, int? H = null, int? W = null)
            {
                return GetOnesParams(N, C, H, W, dtype.CupyDtype.ToString());
            }

            [Test]
            public void Test_Type1()
            {
                int N = 8, C = 3;
                var (x, gamma, beta, mean, var) = GetParams(N, C);
                var y = BatchNorm.Invoke(x, gamma, beta, mean, var).Item1;
                Assert.That(y[0].Data.dtype, Is.TypeOf(xp.float32.GetType()));
            }

            [Test]
            public void Test_Forward1()
            {
                int N = 8, C = 1;
                var (x, gamma, beta, mean, var) = GetParams(N, C);
                var cy = CF.batch_normalization(x.Data, gamma.Data, beta.Data, running_mean: mean.Data, running_var: var.Data);
                var y = BatchNorm.Invoke(x, gamma, beta, mean, var).Item1;
                Assert.That(Utils.array_allclose(y[0].Data, cy));
            }

            [Test]
            public void Test_Forward2()
            {
                int N = 1, C = 10;
                var (x, gamma, beta, mean, var) = GetParams(N, C);
                var cy = CF.batch_normalization(x.Data, gamma.Data, beta.Data, running_mean: mean.Data, running_var: var.Data);
                var y = BatchNorm.Invoke(x, gamma, beta, mean, var).Item1;
                Assert.That(Utils.array_allclose(y[0].Data, cy));
            }

            [Test]
            public void Test_Forward3()
            {
                int N = 20, C = 10;
                var (x, gamma, beta, mean, var) = GetParams(N, C);
                var cy = CF.batch_normalization(x.Data, gamma.Data, beta.Data, running_mean: mean.Data, running_var: var.Data);
                var y = BatchNorm.Invoke(x, gamma, beta, mean, var).Item1;
                Assert.That(Utils.array_allclose(y[0].Data, cy));
            }

            [Test]
            public void Test_Forward4()
            {
                int N = 20, C = 10, H = 5, W = 5;
                var (x, gamma, beta, mean, var) = GetParams(N, C, H, W);
                var cy = CF.batch_normalization(x.Data, gamma.Data, beta.Data, running_mean: mean.Data, running_var: var.Data);
                var y = BatchNorm.Invoke(x, gamma, beta, mean, var).Item1;
                Assert.That(Utils.array_allclose(y[0].Data, cy));
            }

            [Test]
            public void Test_Forward5()
            {
                int N = 20, C = 10;
                var cl = new Tests.Chainer.BatchNormalization(C);
                var l = new Layers.BatchNorm();

                foreach (int i in Enumerable.Range(1, 10))
                {
                    var x = xp.random.randn(N, C).astype("f");
                    var cy = cl.F(x);
                    var y = l.F([x.ToVariable()])[0];
                    Assert.That(Utils.array_allclose(y.Data, cy));
                }

                Assert.That(Utils.array_allclose(cl.avg_mean, l.AvgMean.Data));
                Assert.That(Utils.array_allclose(cl.avg_var, l.AvgVar.Data));
            }

            [Test]
            public void Test_Forward6()
            {
                int N = 20, C = 10, H = 5, W = 5;
                var cl = new Tests.Chainer.BatchNormalization(C);
                var l = new Layers.BatchNorm();

                foreach (int i in Enumerable.Range(1, 10))
                {
                    var x = xp.random.randn(N, C, H, W).astype("f");
                    var cy = cl.F(x);
                    var y = l.F([x.ToVariable()])[0];
                    Assert.That(Utils.array_allclose(y.Data, cy));
                }

                Assert.That(Utils.array_allclose(cl.avg_mean, l.AvgMean.Data));
                Assert.That(Utils.array_allclose(cl.avg_var, l.AvgVar.Data));
            }

            [Test]
            public void Test_Backward1()
            {
                int N = 8, C = 3;
                var (x, gamma, beta, mean, var) = GetOnesParams(dtype: xp.float64, N, C);
                BatchNorm bn = new BatchNorm()
                {
                    AvgMean = mean,
                    InitAvgMean = mean,
                    AvgVar = var,
                    InitAvgVar = var,
                    Decay = 0.9,
                    Eps = 2e-5,
                };
                bn.f = x => BatchNorm.Invoke(bn, x.Get<Variable>("x"), gamma, beta, mean, var);
                Assert.That(Utils.gradient_check(bn, Params<Variable>.args(x), Params<Variable, Variable, Variable, Variable>.args(gamma, beta, mean, var)));
            }

            [Test]
            public void Test_Backward2()
            {
                int N = 8, C = 3;
                var (x, gamma, beta, mean, var) = GetOnesParams(dtype: xp.float64, N, C);
                BatchNorm bn = new BatchNorm()
                {
                    AvgMean = mean,
                    InitAvgMean = mean,
                    AvgVar = var,
                    InitAvgVar = var,
                    Decay = 0.9,
                    Eps = 2e-5,
                };
                bn.f = gamma => BatchNorm.Invoke(bn, x, gamma.Get<Variable>("gamma"), beta, mean, var);
                Assert.That(Utils.gradient_check(bn, Params<Variable>.args(gamma), kwargs: Params<Variable, Variable, Variable, Variable, Variable>.args(x, gamma, beta, mean, var)));
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


            private (Variable, Variable, Variable, Variable, Variable) GetParams(int N, int C, int? H = null, int? W = null,
                string dtype = "f")
            {
                var _dtype = Extensions.dtype(dtype);
                NDarray x, gamma, beta, mean, @var;
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
                @var = xp.abs(xp.random.randn(C).astype(_dtype));

                return (x.ToVariable(), gamma.ToVariable(), beta.ToVariable(), mean.ToVariable(), @var.ToVariable());
            }

            private (Variable, Variable, Variable, Variable, Variable) GetParams(Dtype dtype, int N, int C, int? H = null, int? W = null)
            {
                return GetParams(N, C, H, W, dtype.NumpyDtype.ToString());
            }

            private (Variable, Variable, Variable, Variable, Variable) GetOnesParams(int N, int C, int? H = null, int? W = null,
                string dtype = "f")
            {
                var _dtype = Extensions.dtype(dtype);
                NDarray x, gamma, beta, mean, @var;
                if (H is not null)
                {
                    x = xp.ones(new Shape(N, C, H.Value, W.Value)).astype(_dtype);
                }
                else
                {
                    x = xp.ones(new Shape(N, C)).astype(_dtype);
                }

                gamma = (xp.ones(C) * 2).astype(_dtype);
                beta = xp.ones(C).astype(_dtype);
                mean = xp.ones(C).astype(_dtype);
                @var = xp.ones(C).astype(_dtype);

                return (x.ToVariable(), gamma.ToVariable(), beta.ToVariable(), mean.ToVariable(), @var.ToVariable());
            }

            private (Variable, Variable, Variable, Variable, Variable) GetOnesParams(Dtype dtype, int N, int C, int? H = null, int? W = null)
            {
                return GetOnesParams(N, C, H, W, dtype.NumpyDtype.ToString());
            }

            [Test]
            public void Test_Type1()
            {
                int N = 8, C = 3;
                var (x, gamma, beta, mean, var) = GetParams(N, C);
                var y = BatchNorm.Invoke(x, gamma, beta, mean, var).Item1;
                Assert.That(y[0].Data.dtype, Is.TypeOf(xp.float32.GetType()));
            }

            [Test]
            public void Test_Forward1()
            {
                int N = 8, C = 1;
                var (x, gamma, beta, mean, var) = GetParams(N, C);
                var cy = CF.batch_normalization(x.Data, gamma.Data, beta.Data, running_mean: mean.Data, running_var: var.Data);
                var y = BatchNorm.Invoke(x, gamma, beta, mean, var).Item1;
                Assert.That(Utils.array_allclose(y[0].Data, cy));
            }

            [Test]
            public void Test_Forward2()
            {
                int N = 1, C = 10;
                var (x, gamma, beta, mean, var) = GetParams(N, C);
                var cy = CF.batch_normalization(x.Data, gamma.Data, beta.Data, running_mean: mean.Data, running_var: var.Data);
                var y = BatchNorm.Invoke(x, gamma, beta, mean, var).Item1;
                Assert.That(Utils.array_allclose(y[0].Data, cy));
            }

            [Test]
            public void Test_Forward3()
            {
                int N = 20, C = 10;
                var (x, gamma, beta, mean, var) = GetParams(N, C);
                var cy = CF.batch_normalization(x.Data, gamma.Data, beta.Data, running_mean: mean.Data, running_var: var.Data);
                var y = BatchNorm.Invoke(x, gamma, beta, mean, var).Item1;
                Assert.That(Utils.array_allclose(y[0].Data, cy));
            }

            [Test]
            public void Test_Forward4()
            {
                int N = 20, C = 10, H = 5, W = 5;
                var (x, gamma, beta, mean, var) = GetParams(N, C, H, W);
                var cy = CF.batch_normalization(x.Data, gamma.Data, beta.Data, running_mean: mean.Data, running_var: var.Data);
                var y = BatchNorm.Invoke(x, gamma, beta, mean, var).Item1;
                Assert.That(Utils.array_allclose(y[0].Data, cy));
            }

            [Test]
            public void Test_Forward5()
            {
                int N = 20, C = 10;
                var cl = new Tests.Chainer.BatchNormalization(C);
                var l = new Layers.BatchNorm();

                foreach (int i in Enumerable.Range(1, 10))
                {
                    var x = xp.random.randn(N, C).astype("f");
                    var cy = cl.F(x);
                    var y = l.F([x.ToVariable()])[0];
                    Assert.That(Utils.array_allclose(y.Data, cy));
                }

                Assert.That(Utils.array_allclose(cl.avg_mean, l.AvgMean.Data));
                Assert.That(Utils.array_allclose(cl.avg_var, l.AvgVar.Data));
            }

            [Test]
            public void Test_Forward6()
            {
                int N = 20, C = 10, H = 5, W = 5;
                var cl = new Tests.Chainer.BatchNormalization(C);
                var l = new Layers.BatchNorm();

                foreach (int i in Enumerable.Range(1, 10))
                {
                    var x = xp.random.randn(N, C, H, W).astype("f");
                    var cy = cl.F(x);
                    var y = l.F([x.ToVariable()])[0];
                    Assert.That(Utils.array_allclose(y.Data, cy));
                }

                Assert.That(Utils.array_allclose(cl.avg_mean, l.AvgMean.Data));
                Assert.That(Utils.array_allclose(cl.avg_var, l.AvgVar.Data));
            }

            [Test]
            public void Test_Backward1()
            {
                int N = 8, C = 3;
                var (x, gamma, beta, mean, var) = GetOnesParams(dtype: xp.float64, N, C);
                BatchNorm bn = new BatchNorm()
                {
                    AvgMean = mean,
                    InitAvgMean = mean,
                    AvgVar = var,
                    InitAvgVar = var,
                    Decay = 0.9,
                    Eps = 2e-5,
                };
                bn.f = x => BatchNorm.Invoke(bn, x.Get<Variable>("x"), gamma, beta, mean, var);
                Assert.That(Utils.gradient_check(bn, Params<Variable>.args(x), Params<Variable, Variable, Variable, Variable>.args(gamma, beta, mean, var)));
            }

            [Test]
            public void Test_Backward2()
            {
                int N = 8, C = 3;
                var (x, gamma, beta, mean, var) = GetOnesParams(dtype: xp.float64, N, C);
                BatchNorm bn = new BatchNorm()
                {
                    AvgMean = mean,
                    InitAvgMean = mean,
                    AvgVar = var,
                    InitAvgVar = var,
                    Decay = 0.9,
                    Eps = 2e-5,
                };
                bn.f = gamma => BatchNorm.Invoke(bn, x, gamma.Get<Variable>("gamma"), beta, mean, var);
                Assert.That(Utils.gradient_check(bn, Params<Variable>.args(gamma), kwargs: Params<Variable, Variable, Variable, Variable, Variable>.args(x, gamma, beta, mean, var)));
            }
        }
    }
}
