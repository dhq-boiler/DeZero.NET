using Python.Runtime;

namespace DeZero.NET.Tests
{
    public class AddTests
    {
        [OneTimeSetUp]
        public void OneTimeSetUp()
        {
            Runtime.PythonDLL = @"C:\Users\boiler\AppData\Local\Programs\Python\Python311\python311.dll";
            PythonEngine.Initialize();
            Gpu.Use = false;
        }

        [Test]
        public void Test_Forward1()
        {
            var x0 = xp.array([1, 2, 3]);
            var x1 = new Variable(xp.array([1, 2, 3]));
            var y = x0 + x1;
            var res = y.Data;
            var expected = xp.array([2, 4, 6]);
            Assert.That(res, Is.EqualTo(expected));
        }

        [Test]
        public void Test_DataType()
        {
            var x = new Variable(xp.array(2.0));
            var y = x.pow(2);
            Assert.That(xp.isscalar(y.Data.Array), Is.False);
        }

        [Test]
        public void Test_Backward1()
        {
            var x = xp.random.randn(3, 3).ToVariable();
            var y = xp.random.randn(3, 3).ToVariable();
            Func<Variable[], Variable[]> f = x => [x[0] + y];
            Assert.That(Utils.gradient_check(new Function(f), x), Is.True);
        }

        [Test]
        public void Test_Backward2()
        {
            var x = xp.random.randn(3, 3).ToVariable();
            var y = xp.random.randn(3, 1).ToVariable();
            Func<Variable[], Variable[]> f = x => [x[0] + y];
            Assert.That(Utils.gradient_check(new Function(f), x), Is.True);
        }

        [Test]
        public void Test_Backward3()
        {
            var x = xp.random.randn(3, 3).ToVariable();
            var y = xp.random.randn(3, 1).ToVariable();
            Assert.That(Utils.gradient_check(new Functions.Add(), x, args:y), Is.True);
        }
    }
}