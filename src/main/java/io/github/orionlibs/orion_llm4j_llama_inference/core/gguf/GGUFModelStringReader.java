package io.github.orionlibs.orion_llm4j_llama_inference.core.gguf;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;

final class GGUFModelStringReader
{
    private GGUFModelPrimitiveReader primitiveReader;


    GGUFModelStringReader()
    {
        this.primitiveReader = new GGUFModelPrimitiveReader();
    }


    String readString(FileChannel fileChannel) throws IOException
    {
        int len = Math.toIntExact(primitiveReader.readLong(fileChannel));
        byte[] bytes = new byte[len];
        int bytesRead = fileChannel.read(ByteBuffer.wrap(bytes));
        assert len == bytesRead;
        return new String(bytes, StandardCharsets.UTF_8);
    }
}
