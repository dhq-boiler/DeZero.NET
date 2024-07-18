using DeZero.NET.matplotlib;
using DeZero.NET.Transforms;
using System.IO.Compression;

namespace DeZero.NET.Datasets
{
    public class MNIST : Dataset
    {
        public MNIST(bool train = true, Transform transform = null, Transform target_transform = null)
            : base(train, new Compose([new Flatten(), new ToFloat(), new Normalize(0f, 255f)]), target_transform)
        {
        }

        /// <summary>
        /// C:\Users\USERNAME\.dezero に以下のファイルを配置してください。
        /// * train-images-idx3-ubyte.gz
        /// * train-labels-idx1-ubyte.gz
        /// * t10k-images-idx3-ubyte.gz
        /// * t10k-labels-idx1-ubyte.gz
        /// 以上のファイルは、http://yann.lecun.com/exdb/mnist/ からダウンロードできるはずですが
        /// サーバーの設定不備により、現在ダウンロードできません。
        /// 代替方法として、KaggleのMNISTデータセットをダウンロードして、上記のファイルを配置してください。
        /// </summary>
        public override void Prepare()
        {
            var url = "http://yann.lecun.com/exdb/mnist/";
            Dictionary<string, string> train_files = new();
            train_files.Add("label", "train-labels-idx1-ubyte.gz");
            train_files.Add("target", "train-images-idx3-ubyte.gz");
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

        //gzファイルの先頭16バイトはヘッダ情報なので読み飛ばす
        //17バイト以降からデータが始まる
        //MemoryStreamでデータを受け取って、xp.frombufferでNDarrayに変換
        private NDarray _load_data(string dataPath)
        {
            using (var fs = new FileStream(dataPath, FileMode.Open))
            using (var f = new GZipStream(fs, CompressionMode.Decompress))
            using (var mem = new MemoryStream())
            {
                for (int i = 0; i < 16; i++)
                {
                    f.ReadByte();
                }
                byte[] buffer = new byte[4096]; // 一時的なバッファのサイズを指定します
                int bytesRead;
                while ((bytesRead = f.Read(buffer, 0, buffer.Length)) > 0)
                {
                    mem.Write(buffer, 0, bytesRead);
                }

                var memArr = mem.ToArray();
                using var data = xp.frombuffer(memArr, xp.uint8);
                var data2 = data.reshape(-1, 1, 28, 28);
                return data2;
            }
        }

        private NDarray _load_label(string labelPath)
        {
            using (var fs = new FileStream(labelPath, FileMode.Open))
            using (var f = new GZipStream(fs, CompressionMode.Decompress))
            using (var mem = new MemoryStream())
            {
                for (int i = 0; i < 8; i++)
                {
                    f.ReadByte();
                }
                byte[] buffer = new byte[4096]; // 一時的なバッファのサイズを指定します
                int bytesRead;
                while ((bytesRead = f.Read(buffer, 0, buffer.Length)) > 0)
                {
                    mem.Write(buffer, 0, bytesRead);
                }
                var memArr = mem.ToArray();
                var label = xp.frombuffer(memArr, xp.uint8);
                return label;
            }
        }
    }
}
