package io.github.orionlibs.orion_llm4j_llama_inference.core.gguf;

import io.github.orionlibs.orion_llm4j_inference.core.utils.MetadataValueType;
import java.io.IOException;
import java.nio.channels.FileChannel;

final class GGUFModelMetadataValueTypeReader
{
    private GGUFModelPrimitiveReader primitiveReader;


    GGUFModelMetadataValueTypeReader()
    {
        this.primitiveReader = new GGUFModelPrimitiveReader();
    }


    MetadataValueType readMetadataValueType(FileChannel fileChannel) throws IOException
    {
        int index = primitiveReader.readInt(fileChannel);
        return MetadataValueType.fromIndex(index);
    }
}
