using System.Text.Json;

namespace DeZero.NET.Optimizers.Converters
{
    internal class KeyValuePairJsonConverter : System.Text.Json.Serialization.JsonConverter<KeyValuePair<string, Variable>>
    {
        public override KeyValuePair<string, Variable> Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            string key = null;
            Variable value = null;

            while (reader.Read())
            {
                if (reader.TokenType == JsonTokenType.EndObject)
                {
                    break;
                }

                if (reader.TokenType == JsonTokenType.PropertyName)
                {
                    string propertyName = reader.GetString();
                    reader.Read();

                    switch (propertyName)
                    {
                        case "Key":
                            key = reader.GetString();
                            break;
                        case "Title":
                            //value = new Variable(reader.GetInt32());
                            break;
                    }
                }
            }

            return new KeyValuePair<string, Variable>(key, value);
        }

        public override void Write(Utf8JsonWriter writer, KeyValuePair<string, Variable> value, JsonSerializerOptions options)
        {
            writer.WriteStartObject();

            writer.WriteString("Key", value.Key);
            writer.WriteNumber("Title", value.Value.Title);

            writer.WriteEndObject();
        }
    }
}
