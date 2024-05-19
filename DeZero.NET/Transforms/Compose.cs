using DeZero.NET.PIL;

namespace DeZero.NET.Transforms
{
    public class Compose : Transform
    {
        public Transform[] Transforms { get; set; }

        public Compose(params Transform[] transforms)
        {
            this.Transforms = transforms;
        }

        public override T Call<T>(PythonObject obj)
        {
            return InternalCall<T>(obj);
        }

        public override Image ToImage(Image image)
        {
            if (!Transforms.Any())
                return image;

            object obj = image;
            foreach (var t in Transforms)
            {
                switch (t.GetType().Name)
                {
                    case "AsType":
                        obj = t.ToNDarray(obj as NDarray);
                        break;
                    case "CenterCrop":
                        obj = t.ToImage(obj as Image);
                        break;
                    case "Compose":
                        obj = t.ToImage(obj as Image);
                        break;
                    case "Convert":
                        obj = t.ToImage(obj as Image);
                        break;
                    case "Flatten":
                        obj = t.ToNDarray(obj as NDarray);
                        break;
                    case "Normalize":
                        obj = t.ToNDarray(obj as NDarray);
                        break;
                    case "Resize":
                        obj = t.ToImage(obj as Image);
                        break;
                    case "ToArray":
                        obj = t.ToNDarray(obj as PythonObject);
                        break;
                    case "ToPIL":
                        obj = t.ToImage(obj as NDarray);
                        break;
                    case "ToFloat":
                        obj = t.ToNDarray(obj as NDarray);
                        break;
                    case "ToInt":
                        obj = t.ToNDarray(obj as NDarray);
                        break;
                }
            }

            return obj as Image;
        }

        public override Image ToImage(NDarray array)
        {
            if (!Transforms.Any())
                throw new InvalidOperationException();

            object obj = array;
            foreach (var t in Transforms)
            {
                switch (t.GetType().Name)
                {
                    case "AsType":
                        obj = t.ToNDarray(obj as NDarray);
                        break;
                    case "CenterCrop":
                        obj = t.ToImage(obj as Image);
                        break;
                    case "Compose":
                        obj = t.ToImage(obj as Image);
                        break;
                    case "Convert":
                        obj = t.ToImage(obj as Image);
                        break;
                    case "Flatten":
                        obj = t.ToNDarray(obj as NDarray);
                        break;
                    case "Normalize":
                        obj = t.ToNDarray(obj as NDarray);
                        break;
                    case "Resize":
                        obj = t.ToImage(obj as Image);
                        break;
                    case "ToArray":
                        obj = t.ToNDarray(obj as PythonObject);
                        break;
                    case "ToPIL":
                        obj = t.ToImage(obj as NDarray);
                        break;
                    case "ToFloat":
                        obj = t.ToNDarray(obj as NDarray);
                        break;
                    case "ToInt":
                        obj = t.ToNDarray(obj as NDarray);
                        break;
                }
            }

            return obj as Image;
        }

        public override NDarray ToNDarray(PythonObject arg)
        {
            if (!Transforms.Any())
                return arg as NDarray;

            object obj = arg;
            foreach (var t in Transforms)
            {
                switch (t.GetType().Name)
                {
                    case "AsType":
                        obj = t.ToNDarray(obj as NDarray);
                        break;
                    case "CenterCrop":
                        obj = t.ToImage(obj as Image);
                        break;
                    case "Compose":
                        obj = t.ToNDarray(obj as Image);
                        break;
                    case "Convert":
                        obj = t.ToImage(obj as Image);
                        break;
                    case "Flatten":
                        obj = t.ToNDarray(obj as NDarray);
                        break;
                    case "Normalize":
                        obj = t.ToNDarray(obj as NDarray);
                        break;
                    case "Resize":
                        obj = t.ToImage(obj as Image);
                        break;
                    case "ToArray":
                        obj = t.ToNDarray(obj as PythonObject);
                        break;
                    case "ToPIL":
                        obj = t.ToImage(obj as NDarray);
                        break;
                    case "ToFloat":
                        obj = t.ToNDarray(obj as NDarray);
                        break;
                    case "ToInt":
                        obj = t.ToNDarray(obj as NDarray);
                        break;
                }
            }

            return obj as NDarray;
        }

        public override NDarray ToNDarray(NDarray array)
        {
            if (!Transforms.Any())
                return array;

            object obj = array;
            foreach (var t in Transforms)
            {
                switch (t.GetType().Name)
                {
                    case "AsType":
                        obj = t.ToNDarray(obj as NDarray);
                        break;
                    case "CenterCrop":
                        obj = t.ToImage(obj as Image);
                        break;
                    case "Compose":
                        obj = t.ToNDarray(obj as Image);
                        break;
                    case "Convert":
                        obj = t.ToImage(obj as Image);
                        break;
                    case "Flatten":
                        obj = t.ToNDarray(obj as NDarray);
                        break;
                    case "Normalize":
                        obj = t.ToNDarray(obj as NDarray);
                        break;
                    case "Resize":
                        obj = t.ToImage(obj as Image);
                        break;
                    case "ToArray":
                        obj = t.ToNDarray(obj as PythonObject);
                        break;
                    case "ToPIL":
                        obj = t.ToImage(obj as NDarray);
                        break;
                    case "ToFloat":
                        obj = t.ToNDarray(obj as NDarray);
                        break;
                    case "ToInt":
                        obj = t.ToNDarray(obj as NDarray);
                        break;
                }
            }

            return obj as NDarray;
        }
    }
}
