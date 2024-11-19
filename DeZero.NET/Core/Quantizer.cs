namespace DeZero.NET.Core
{
    public class Quantizer
    {
        private readonly int _bits;
        private readonly float _scale;
        private readonly float _zeroPoint;

        public Quantizer(int bits)
        {
            _bits = bits;
            _scale = (float)(Math.Pow(2, bits) - 1);
            _zeroPoint = _scale / 2;
        }

        public NDarray Quantize(NDarray input)
        {
            var normalized = (input + _zeroPoint) / _scale;
            var quantized = xp.clip(xp.round(normalized * _scale), new NDarray(0), new NDarray(_scale));
            return (quantized / _scale) * _scale - _zeroPoint;
        }
    }
}
