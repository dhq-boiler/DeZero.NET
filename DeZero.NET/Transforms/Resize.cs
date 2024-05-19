using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DeZero.NET.PIL;

namespace DeZero.NET.Transforms
{
    public class Resize : Transform
    {
        public (int, int) Size { get; private set; }
        public Image.Mode Mode { get; set; }

        public Resize(int size, Image.Mode mode = Image.Mode.BILINEAR)
        {
            Size = (size, size);
            Mode = mode;
        }

        public Resize((int, int) size, Image.Mode mode = Image.Mode.BILINEAR)
        {
            Size = size;
            Mode = mode;
        }

        public override T Call<T>(PythonObject obj)
        {
            return InternalCall<T>(obj);
        }

        public override Image ToImage(Image image)
        {
            return image.resize(Size, Mode);
        }

        public override Image ToImage(NDarray array)
        {
            throw new NotSupportedException();
        }

        public override NDarray ToNDarray(PythonObject obj)
        {
            throw new NotSupportedException();
        }

        public override NDarray ToNDarray(NDarray array)
        {
            throw new NotSupportedException();
        }
    }
}
