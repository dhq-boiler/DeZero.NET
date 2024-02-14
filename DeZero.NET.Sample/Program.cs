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

//画像の表示
var image = PILImage.open(@"C:\Users\boiler\Desktop\ScreenShot-5.jpg");
var imageArr = xp.array(image);
pyplot.imshow(imageArr);
pyplot.show();

//折れ線グラフの表示
var b1 = xp.linspace(0, 10, 100);
var b2 = b1 + xp.random.randn(100) * 100;
var c1 = xp.linspace(0, 10, 100);
var c2 = c1 + xp.random.randn(100) * 50;

var line1 = pyplot.plot(b1, b2, "b-", label: "test1");
var line2 = pyplot.plot(c1, c2, "r-", label: "test2");
pyplot.title("Sample Plot");
pyplot.xlabel("X Axis Label");
pyplot.ylabel("Y Axis Label");
pyplot.legend(handles: [line1[0], line2[0]], ["Test1", "Test2"]);
pyplot.show();