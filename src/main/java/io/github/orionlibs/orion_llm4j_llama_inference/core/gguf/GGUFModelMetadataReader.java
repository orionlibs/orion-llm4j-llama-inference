package io.github.orionlibs.orion_llm4j_llama_inference.core.gguf;

import io.github.orionlibs.orion_llm4j_llama_inference.core.utils.MetadataValueType;
import java.io.IOException;
import java.nio.channels.FileChannel;

final class GGUFModelMetadataReader
{
    private GGUFModelArrayReader arrayReader;
    private GGUFModelPrimitiveReader primitiveReader;
    private GGUFModelStringReader stringReader;
    private GGUFModelMetadataValueTypeReader metadataValueTypeReader;


    GGUFModelMetadataReader()
    {
        this.arrayReader = new GGUFModelArrayReader();
        this.primitiveReader = new GGUFModelPrimitiveReader();
        this.stringReader = new GGUFModelStringReader();
        this.metadataValueTypeReader = new GGUFModelMetadataValueTypeReader();
    }


    Object readMetadataValue(FileChannel fileChannel) throws IOException
    {
        MetadataValueType value_type = metadataValueTypeReader.readMetadataValueType(fileChannel);
        return readMetadataValueBasedOnType(value_type, fileChannel);
    }


    private Object readMetadataValueBasedOnType(MetadataValueType valueType, FileChannel fileChannel) throws IOException
    {
        return switch(valueType)
        {
            case UINT8, INT8 -> primitiveReader.readByte(fileChannel);
            case UINT16, INT16 -> primitiveReader.readShort(fileChannel);
            case UINT32, INT32 -> primitiveReader.readInt(fileChannel);
            case FLOAT32 -> primitiveReader.readFloat(fileChannel);
            case UINT64, INT64 -> primitiveReader.readLong(fileChannel);
            case FLOAT64 -> primitiveReader.readDouble(fileChannel);
            case BOOL -> primitiveReader.readBoolean(fileChannel);
            case STRING -> stringReader.readString(fileChannel);
            case ARRAY -> arrayReader.readArray(fileChannel);
        };
    }
}
