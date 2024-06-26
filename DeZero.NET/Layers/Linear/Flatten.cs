﻿namespace DeZero.NET.Layers.Linear
{
    public class Flatten : Layer
    {
        public override Variable[] Forward(params Variable[] xs)
        {
            return [Functions.Flatten.Invoke(xs[0])];
        }
    }
}
