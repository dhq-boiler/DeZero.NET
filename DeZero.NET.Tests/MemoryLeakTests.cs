using DeZero.NET.Core;
using Python.Runtime;

namespace DeZero.NET.Tests
{
    [Category("cupy")]
    public class MemoryLeakTests
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
        public void MemoryLeakTest()
        {
            //巨大なNDarrayを生成
            var x = xp.random.randn(1000, 1000, 1000);

            //メモリ使用量が大きいはず
            using var gpuInfo1 = new GpuMemoryInfo();
            Assert.That(gpuInfo1.UsedMemoryMB, Is.EqualTo(7629L));

            Thread.Sleep(1000);

            //巨大なNDarrayを解放
            x.Dispose();
            
            //メモリ使用量が全くないはず
            using var gpuInfo2 = new GpuMemoryInfo();
            Assert.That(gpuInfo2.UsedMemoryMB, Is.EqualTo(0L));
        }
    }
}
