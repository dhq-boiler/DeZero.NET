using DeZero.NET;
using DeZero.NET.matplotlib;
using DeZero.NET.PIL;
using Python.Runtime;

Runtime.PythonDLL = @"C:\Users\boiler\AppData\Local\Programs\Python\Python311\python311.dll";

//Enable GPU
Core.UseGpu = true;

// before starting the measurement, let us call Numpy/CuPy once to get the setup checks done. 
xp.arange(1);

var a1 = xp.arange(60000).reshape(300, 200);
var a2 = xp.arange(80000).reshape(200, 400);

var result = xp.matmul(a1, a2);

Console.WriteLine(result.repr);

var image = PILImage.open(@"C:\Users\boiler\Desktop\ScreenShot-5.jpg");
var imageArr = xp.array(image);
pyplot.imshow(imageArr);
pyplot.show();