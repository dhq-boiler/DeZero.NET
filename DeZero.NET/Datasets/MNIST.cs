using DeZero.NET.Transforms;
using System.IO.Compression;
using DeZero.NET.matplotlib;

namespace DeZero.NET.Datasets
{
    public class MNIST : Dataset
    {
        public MNIST(bool train = true, Transform transform = null, Transform target_transform = null)
            : base(train, new Compose([new Flatten(), new ToFloat(), new Normalize(new NDarray(0f), new NDarray(255f))]), target_transform)
        {
        }

        public override void Prepare()
        {
            var url = "http://yann.lecun.com/exdb/mnist/";
            Dictionary<string, string> train_files = new();
            train_files.Add("target", "train-images-idx3-ubyte.gz");
            train_files.Add("label", "train-labels-idx1-ubyte.gz");
            Dictionary<string, string> test_files = new();
            test_files.Add("target", "t10k-images-idx3-ubyte.gz");
            test_files.Add("label", "t10k-labels-idx1-ubyte.gz");

            var files = Train ? train_files : test_files;
            var data_path = Utils.get_file(url + files["target"]);
            var label_path = Utils.get_file(url + files["label"]);

            Data = _load_data(data_path);
            Label = _load_label(label_path);
        }

        public void Show(int row = 10, int col = 10)
        {
            int H = 28, W = 28;
            var img = xp.zeros(new Shape(H * row, W * col));
            for (int r = 0; r < row; r++)
            {
                for (int c = 0; c < col; c++)
                {
                    var i = r * col + c;
                    var x = Data[xp.random.randint(0, Data.len - 1)].reshape(H, W);
                    img[new Slice(r * H, (r + 1) * H), new Slice(c * W, (c + 1) * W)] = x;
                }
            }
            pyplot.imshow(img, cmap:"gray", interpolation:"nearest");
            pyplot.axis("off");
            pyplot.show();
        }

        private NDarray _load_data(string dataPath)
        {
            using (var fs = new FileStream(dataPath, FileMode.Open))
            using (var f = new GZipStream(fs, CompressionMode.Decompress))
            {
                var magic = Utils.read_int(f);
                var num = Utils.read_int(f);
                var rows = Utils.read_int(f);
                var cols = Utils.read_int(f);
                var size = num * rows * cols;
                var data = new NDarray(size);
                for (int i = 0; i < size; i++)
                {
                    data[i] = new NDarray(Utils.read_byte(f));
                }
                return data.reshape(num, rows, cols);
            }
        }

        private NDarray _load_label(string labelPath)
        {
            using (var fs = new FileStream(labelPath, FileMode.Open))
            using (var f = new GZipStream(fs, CompressionMode.Decompress))
            {
                var magic = Utils.read_int(f);
                var num = Utils.read_int(f);
                var size = num;
                var label = new NDarray(size);
                for (int i = 0; i < size; i++)
                {
                    label[i] = new NDarray(Utils.read_byte(f));
                }
                return label;
            }
        }
    }
}
