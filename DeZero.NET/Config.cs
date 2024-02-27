namespace DeZero.NET
{
    public static class Config
    {
        public static bool EnableBackprop { get; set; } = true;
        public static bool Train { get; set; } = true;
    }

    public class UsingConfig : IDisposable
    {
        private string _name;
        private bool _originalValue;

        public UsingConfig(string name, bool value)
        {
            _name = name;
            _originalValue = name switch
            {
                "EnableBackprop" => Config.EnableBackprop,
                "Train" => Config.Train,
                _ => throw new ArgumentException("Invalid configuration name.")
            };

            switch (name)
            {
                case "EnableBackprop":
                    Config.EnableBackprop = value;
                    break;
                case "Train":
                    Config.Train = value;
                    break;
            }
        }

        public void Dispose()
        {
            switch (_name)
            {
                case "EnableBackprop":
                    Config.EnableBackprop = _originalValue;
                    break;
                case "Train":
                    Config.Train = _originalValue;
                    break;
            }
        }
    }

    public static class ConfigExtensions
    {
        public static IDisposable NoGrad()
        {
            return new UsingConfig("EnableBackprop", false);
        }

        public static IDisposable TestMode()
        {
            return new UsingConfig("Train", false);
        }
    }
}
