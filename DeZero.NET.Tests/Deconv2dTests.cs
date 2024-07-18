using DeZero.NET.Core;
using DeZero.NET.Functions;
using Python.Runtime;

namespace DeZero.NET.Tests
{
    public class Deconv2dTests
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
                int n = 10, c_i = 1, c_o = 3;
                int h_i = 5, w_i = 10;
                int h_k = 10, w_k = 10;
                int h_p = 5, w_p = 5;
                int s_y = 5, s_x = 5;
                var x = xp.random.uniform(new NDarray<float>(0), new NDarray<float>(1), [n, c_i, h_i, w_i]).astype(xp.float32).ToVariable();
                var W = xp.random.uniform(new NDarray<float>(0), new NDarray<float>(1), [c_i, c_o, h_k, w_k]).astype(xp.float32).ToVariable();
                var b = xp.random.uniform(new NDarray<float>(0), new NDarray<float>(1), [c_o]).astype(xp.float32).ToVariable();
                var expected = Chainer.CF.deconvolution_2d(x.Data.Value, W.Data.Value, b.Data.Value, stride: (s_y, s_x), pad: (h_p, w_p));
                var y = Deconv2d.Invoke(x, W, b, stride: (s_y, s_x), pad: (h_p, w_p));
                Assert.IsTrue(Utils.array_allclose(expected, y[0].Data.Value));
            }

            [Test]
            public void Test_Forward2()
            {
                int n = 10, c_i = 1, c_o = 3;
                int h_i = 5, w_i = 10;
                int h_k = 10, w_k = 10;
                int h_p = 5, w_p = 5;
                int s_y = 5, s_x = 5;
                var x = xp.random.uniform(new NDarray<float>(0), new NDarray<float>(1), [n, c_i, h_i, w_i]).astype(xp.float32).ToVariable();
                var W = xp.random.uniform(new NDarray<float>(0), new NDarray<float>(1), [c_i, c_o, h_k, w_k]).astype(xp.float32).ToVariable();
                Variable b = null;
                var expected = Chainer.CF.deconvolution_2d(x.Data.Value, W.Data.Value, b?.Data.Value, stride: (s_y, s_x), pad: (h_p, w_p));
                var y = Deconv2d.Invoke(x, W, b, stride: (s_y, s_x), pad: (h_p, w_p));
                Assert.IsTrue(Utils.array_allclose(expected, y[0].Data.Value));
            }

            [Test]
            public void Test_Backward1()
            {
                int n = 10, c_i = 1, c_o = 3;
                int h_i = 5, w_i = 10;
                int h_k = 10, w_k = 10;
                int h_p = 5, w_p = 5;
                int s_y = 5, s_x = 5;
                var x = xp.random.uniform(new NDarray<float>(0), new NDarray<float>(1), [n, c_i, h_i, w_i]).ToVariable();
                var W = xp.random.uniform(new NDarray<float>(0), new NDarray<float>(1), [c_i, c_o, h_k, w_k]).ToVariable();
                Variable b = null;
                Func<Params, Variable[]> f = args => Deconv2d.Invoke(args.Get<Variable>("x"), W, b, stride: (s_y, s_x), pad: (h_p, w_p));
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetKeywordArg(x)));
            }

            [Test]
            public void Test_Backward2()
            {
                int n = 10, c_i = 1, c_o = 3;
                int h_i = 5, w_i = 10;
                int h_k = 10, w_k = 10;
                int h_p = 5, w_p = 5;
                int s_y = 5, s_x = 5;
                var x = xp.random.uniform(new NDarray<float>(0), new NDarray<float>(1), [n, c_i, h_i, w_i]).ToVariable();
                var W = xp.random.uniform(new NDarray<float>(0), new NDarray<float>(1), [c_i, c_o, h_k, w_k]).ToVariable();
                var b = xp.random.uniform(new NDarray<float>(0), new NDarray<float>(1), [c_o])?.ToVariable();
                var f = new Func<Params, Variable[]>(args => Deconv2d.Invoke(x, args.Get<Variable>("x"), b, stride: (s_y, s_x), pad: (h_p, w_p)));
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetKeywordArg(W)));
            }

            [Test]
            public void Test_Backward3()
            {
                int n = 10, c_i = 1, c_o = 3;
                int h_i = 5, w_i = 10;
                int h_k = 10, w_k = 10;
                int h_p = 5, w_p = 5;
                int s_y = 5, s_x = 5;
                var x = xp.random.uniform(new NDarray<float>(0), new NDarray<float>(1), [n, c_i, h_i, w_i]).ToVariable();
                var W = xp.random.uniform(new NDarray<float>(0), new NDarray<float>(1), [c_i, c_o, h_k, w_k]).ToVariable();
                var b = xp.random.uniform(new NDarray<float>(0), new NDarray<float>(1), [c_o])?.ToVariable();
                var f = new Func<Params, Variable[]>(args => Deconv2d.Invoke(x, W, args.Get<Variable>("x"), stride: (s_y, s_x), pad: (h_p, w_p)));
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetKeywordArg(b)));
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
                int n = 10, c_i = 1, c_o = 3;
                int h_i = 5, w_i = 10;
                int h_k = 10, w_k = 10;
                int h_p = 5, w_p = 5;
                int s_y = 5, s_x = 5;
                var x = xp.random.uniform(new NDarray<float>(0), new NDarray<float>(1), [n, c_i, h_i, w_i]).astype(xp.float32).ToVariable();
                var W = xp.random.uniform(new NDarray<float>(0), new NDarray<float>(1), [c_i, c_o, h_k, w_k]).astype(xp.float32).ToVariable();
                var b = xp.random.uniform(new NDarray<float>(0), new NDarray<float>(1), [c_o]).astype(xp.float32).ToVariable();
                var expected = Chainer.CF.deconvolution_2d(x.Data.Value, W.Data.Value, b.Data.Value, stride: (s_y, s_x), pad: (h_p, w_p));
                var y = Deconv2d.Invoke(x, W, b, stride: (s_y, s_x), pad: (h_p, w_p));
                Assert.IsTrue(Utils.array_allclose(expected, y[0].Data.Value));
            }

            [Test]
            public void Test_Forward2()
            {
                int n = 10, c_i = 1, c_o = 3;
                int h_i = 5, w_i = 10;
                int h_k = 10, w_k = 10;
                int h_p = 5, w_p = 5;
                int s_y = 5, s_x = 5;
                var x = xp.random.uniform(new NDarray<float>(0), new NDarray<float>(1), [n, c_i, h_i, w_i]).astype(xp.float32).ToVariable();
                var W = xp.random.uniform(new NDarray<float>(0), new NDarray<float>(1), [c_i, c_o, h_k, w_k]).astype(xp.float32).ToVariable();
                Variable b = null;
                var expected = Chainer.CF.deconvolution_2d(x.Data.Value, W.Data.Value, b?.Data.Value, stride: (s_y, s_x), pad: (h_p, w_p));
                var y = Deconv2d.Invoke(x, W, b, stride: (s_y, s_x), pad: (h_p, w_p));
                Assert.IsTrue(Utils.array_allclose(expected, y[0].Data.Value));
            }

            [Test]
            public void Test_Backward1()
            {
                int n = 10, c_i = 1, c_o = 3;
                int h_i = 5, w_i = 10;
                int h_k = 10, w_k = 10;
                int h_p = 5, w_p = 5;
                int s_y = 5, s_x = 5;
                var x = xp.random.uniform(new NDarray<float>(0), new NDarray<float>(1), [n, c_i, h_i, w_i]).ToVariable();
                var W = xp.random.uniform(new NDarray<float>(0), new NDarray<float>(1), [c_i, c_o, h_k, w_k]).ToVariable();
                Variable b = null;
                Func<Params, Variable[]> f = args => Deconv2d.Invoke(args.Get<Variable>("x"), W, b, stride: (s_y, s_x), pad: (h_p, w_p));
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetKeywordArg(x)));
            }

            [Test]
            public void Test_Backward2()
            {
                int n = 10, c_i = 1, c_o = 3;
                int h_i = 5, w_i = 10;
                int h_k = 10, w_k = 10;
                int h_p = 5, w_p = 5;
                int s_y = 5, s_x = 5;
                var x = xp.random.uniform(new NDarray<float>(0), new NDarray<float>(1), [n, c_i, h_i, w_i]).ToVariable();
                var W = xp.random.uniform(new NDarray<float>(0), new NDarray<float>(1), [c_i, c_o, h_k, w_k]).ToVariable();
                var b = xp.random.uniform(new NDarray<float>(0), new NDarray<float>(1), [c_o])?.ToVariable();
                var f = new Func<Params, Variable[]>(args => Deconv2d.Invoke(x, args.Get<Variable>("x"), b, stride: (s_y, s_x), pad: (h_p, w_p)));
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetKeywordArg(W)));
            }

            [Test]
            public void Test_Backward3()
            {
                int n = 10, c_i = 1, c_o = 3;
                int h_i = 5, w_i = 10;
                int h_k = 10, w_k = 10;
                int h_p = 5, w_p = 5;
                int s_y = 5, s_x = 5;
                var x = xp.random.uniform(new NDarray<float>(0), new NDarray<float>(1), [n, c_i, h_i, w_i]).ToVariable();
                var W = xp.random.uniform(new NDarray<float>(0), new NDarray<float>(1), [c_i, c_o, h_k, w_k]).ToVariable();
                var b = xp.random.uniform(new NDarray<float>(0), new NDarray<float>(1), [c_o])?.ToVariable();
                var f = new Func<Params, Variable[]>(args => Deconv2d.Invoke(x, W, args.Get<Variable>("x"), stride: (s_y, s_x), pad: (h_p, w_p)));
                Assert.IsTrue(Utils.gradient_check(new Function(f), Params.New.SetKeywordArg(b)));
            }
        }
    }
}
