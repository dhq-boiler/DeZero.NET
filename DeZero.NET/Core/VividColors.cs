namespace DeZero.NET.Core
{
    public static class VividColors
    {
        private static readonly Random _random = new Random();

        // 赤系
        public static readonly string Red = "#FF0000";
        public static readonly string DeepPink = "#FF1493";
        public static readonly string OrangeRed = "#FF4500";
        public static readonly string LightCoral = "#FF6B6B";
        public static readonly string HotPink = "#FF69B4";
        public static readonly string DarkOrange = "#FF8C00";
        public static readonly string Orange = "#FFA500";
        public static readonly string LightPink = "#FFB6C1";
        public static readonly string Coral = "#FF7F50";
        public static readonly string Tomato = "#FF6347";

        // ピンク系
        public static readonly string CherryBlossomPink = "#FFB7C5";
        public static readonly string PalePink = "#FF82AB";
        public static readonly string RosePink = "#FF34B3";
        public static readonly string DeepRose = "#FF1A75";
        public static readonly string Rose = "#FF007F";
        public static readonly string SalmonPink = "#FF91A4";
        public static readonly string MediumPink = "#FF85A6";
        public static readonly string FuchsiaPink = "#FF77FF";

        // オレンジ系
        public static readonly string LightSalmon = "#FFA07A";
        public static readonly string Salmon = "#FF8C69";
        public static readonly string Tan = "#FFA54F";
        public static readonly string DeepOrange = "#FF9933";
        public static readonly string Orange2 = "#FFA500";
        public static readonly string LightOrange = "#FFB347";

        // 黄色系
        public static readonly string Gold = "#FFD700";
        public static readonly string BananaYellow = "#FFE135";
        public static readonly string Yellow = "#FFFF00";
        public static readonly string CanaryYellow = "#FFEF00";
        public static readonly string LemonYellow = "#FFE700";
        public static readonly string SchoolBusYellow = "#FFD800";
        public static readonly string GoldenYellow = "#FFC800";
        public static readonly string SelectiveYellow = "#FFB700";
        public static readonly string OrangeYellow = "#FFA600";
        public static readonly string DarkYellow = "#FF9900";

        // 緑系
        public static readonly string Lime = "#00FF00";
        public static readonly string LimeGreen = "#32CD32";
        public static readonly string PearlAqua = "#98FB98";
        public static readonly string LightGreen = "#90EE90";
        public static readonly string MediumSpringGreen = "#00FA9A";
        public static readonly string SpringGreen = "#00FF7F";
        public static readonly string MediumSeaGreen = "#3CB371";
        public static readonly string SeaGreen = "#2E8B57";
        public static readonly string ForestGreen = "#228B22";
        public static readonly string Green = "#008000";

        // ターコイズ系
        public static readonly string Turquoise = "#40E0D0";
        public static readonly string MediumTurquoise = "#48D1CC";
        public static readonly string DarkTurquoise = "#00CED1";
        public static readonly string LightSeaGreen = "#20B2AA";
        public static readonly string DarkCyan = "#008B8B";
        public static readonly string Teal = "#008080";
        public static readonly string Cyan = "#00FFFF";
        public static readonly string LightCyan = "#E0FFFF";
        public static readonly string PaleTurquoise = "#AFEEEE";
        public static readonly string Aquamarine = "#7FFFD4";

        // 青系
        public static readonly string Blue = "#0000FF";
        public static readonly string MediumBlue = "#0000CD";
        public static readonly string DarkBlue = "#00008B";
        public static readonly string Navy = "#000080";
        public static readonly string RoyalBlue = "#4169E1";
        public static readonly string DodgerBlue = "#1E90FF";
        public static readonly string DeepSkyBlue = "#00BFFF";
        public static readonly string SkyBlue = "#87CEEB";
        public static readonly string LightSkyBlue = "#87CEFA";
        public static readonly string SteelBlue = "#4682B4";

        // 紫系
        public static readonly string Violet = "#8B00FF";
        public static readonly string DarkViolet = "#9400D3";
        public static readonly string DarkOrchid = "#9932CC";
        public static readonly string MediumOrchid = "#BA55D3";
        public static readonly string Orchid = "#DA70D6";
        public static readonly string Magenta = "#FF00FF";
        public static readonly string Plum = "#DDA0DD";

        // ブラウン系
        public static readonly string SaddleBrown = "#8B4513";
        public static readonly string Sienna = "#A0522D";
        public static readonly string Brown = "#A52A2A";
        public static readonly string DarkGoldenrod = "#B8860B";
        public static readonly string Peru = "#CD853F";
        public static readonly string Chocolate = "#D2691E";
        public static readonly string Goldenrod = "#DAA520";
        public static readonly string BurlyWood = "#DEB887";
        public static readonly string SandyBrown = "#F4A460";
        public static readonly string Copper = "#B87333";

        // その他の鮮やかな色
        public static readonly string Fuchsia = "#FF00FF";
        public static readonly string YellowGreen = "#9ACD32";

        // すべての色のコレクション
        public static readonly IReadOnlyList<string> HexColors = new[]
        {
            Red, DeepPink, OrangeRed, LightCoral, HotPink, DarkOrange, Orange, LightPink, Coral, Tomato,
            CherryBlossomPink, PalePink, RosePink, DeepRose, Rose, SalmonPink, MediumPink, FuchsiaPink,
            LightSalmon, Salmon, Tan, DeepOrange, Orange2, LightOrange,
            Gold, BananaYellow, Yellow, CanaryYellow, LemonYellow, SchoolBusYellow, GoldenYellow, SelectiveYellow,
            OrangeYellow, DarkYellow,
            Lime, LimeGreen, PearlAqua, LightGreen, MediumSpringGreen, SpringGreen, MediumSeaGreen, SeaGreen,
            ForestGreen, Green,
            Turquoise, MediumTurquoise, DarkTurquoise, LightSeaGreen, DarkCyan, Teal, Cyan, LightCyan, PaleTurquoise,
            Aquamarine,
            Blue, MediumBlue, DarkBlue, Navy, RoyalBlue, DodgerBlue, DeepSkyBlue, SkyBlue, LightSkyBlue, SteelBlue,
            Violet, DarkViolet, DarkOrchid, MediumOrchid, Orchid, Magenta, Plum,
            SaddleBrown, Sienna, Brown, DarkGoldenrod, Peru, Chocolate, Goldenrod, BurlyWood, SandyBrown, Copper,
            Fuchsia, YellowGreen,
        };

        // RGB値を16進数文字列に変換するユーティリティメソッド
        public static string ToHexString(int r, int g, int b)
        {
            return $"#{r:X2}{g:X2}{b:X2}";
        }

        // 16進数文字列からRGB値を取得するユーティリティメソッド
        public static (int R, int G, int B) FromHexString(string hexColor)
        {
            if (hexColor == null)
                throw new ArgumentNullException(nameof(hexColor));

            hexColor = hexColor.TrimStart('#');
            if (hexColor.Length != 6)
                throw new ArgumentException("Invalid hex color format", nameof(hexColor));

            int r = Convert.ToInt32(hexColor.Substring(0, 2), 16);
            int g = Convert.ToInt32(hexColor.Substring(2, 2), 16);
            int b = Convert.ToInt32(hexColor.Substring(4, 2), 16);

            return (r, g, b);
        }

        /// <summary>
        /// HexColorsからランダムにn個の色を重複ありで取得します
        /// </summary>
        public static IReadOnlyList<string> GetRandomColors(int count)
        {
            if (count < 0 || count > HexColors.Count)
            {
                throw new ArgumentException($"要求された色の数（{count}）が不正です。0から{HexColors.Count}の間で指定してください。");
            }

            var colors = HexColors.ToList();
            var result = new List<string>(count);

            for (int i = 0; i < count; i++)
            {
                int index = _random.Next(i, colors.Count);
                if (i != index)
                {
                    (colors[i], colors[index]) = (colors[index], colors[i]);
                }
                result.Add(colors[i]);
            }

            return result.AsReadOnly();
        }

        /// <summary>
        /// HexColorsからランダムにn個の色を重複なしで取得します
        /// </summary>
        public static IReadOnlyList<string> GetUniqueRandomColors(int count)
        {
            if (count < 0 || count > HexColors.Count)
            {
                throw new ArgumentException($"要求された色の数（{count}）が不正です。0から{HexColors.Count}の間で指定してください。");
            }

            var availableColors = new HashSet<string>(HexColors);
            var result = new List<string>(count);

            while (result.Count < count)
            {
                // HashSetから直接要素を取り出すためにToListを使用
                var currentColor = availableColors.ToList()[_random.Next(availableColors.Count)];
                result.Add(currentColor);
                availableColors.Remove(currentColor);
            }

            return result.AsReadOnly();
        }

        /// <summary>
        /// HexColorsからランダムに1つの色を取得します
        /// </summary>
        public static string GetRandomColor()
        {
            return HexColors[_random.Next(HexColors.Count)];
        }
    }
}
