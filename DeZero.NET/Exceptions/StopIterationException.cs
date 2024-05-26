using System.Runtime.Serialization;

namespace DeZero.NET.Exceptions
{


    [Serializable]
    public class StopIterationException : Exception
    {
        public StopIterationException()
        {
        }

        public StopIterationException(string? message) : base(message)
        {
        }

        public StopIterationException(string? message, Exception? innerException) : base(message, innerException)
        {
        }

        protected StopIterationException(SerializationInfo info, StreamingContext context) : base(info, context)
        {
        }
    }
}
