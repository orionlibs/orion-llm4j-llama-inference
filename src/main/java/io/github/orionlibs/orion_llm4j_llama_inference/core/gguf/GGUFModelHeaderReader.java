package io.github.orionlibs.orion_llm4j_llama_inference.core.gguf;

import io.github.orionlibs.orion_llm4j_inference.core.utils.Pair;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

final class GGUFModelHeaderReader
{
    private static final int GGUF_MAGIC = 0x46554747;
    private static final List<Integer> SUPPORTED_GGUF_VERSIONS = List.of(2, 3);
    private int magic;
    private int tensorCount;
    private Map<String, Object> metadata;
    private int version;
    private int metadata_kv_count;
    private GGUFModelPrimitiveReader primitiveReader;
    private GGUFModelKeyValuePairReader keyValuePairReader;


    GGUFModelHeaderReader()
    {
        this.primitiveReader = new GGUFModelPrimitiveReader();
        this.keyValuePairReader = new GGUFModelKeyValuePairReader();
    }


    void readHeader(FileChannel fileChannel) throws IOException
    {
        this.magic = primitiveReader.readInt(fileChannel);
        if(magic != GGUF_MAGIC)
        {
            throw new IllegalArgumentException("unsupported header.magic " + magic);
        }
        this.version = primitiveReader.readInt(fileChannel);
        if(!SUPPORTED_GGUF_VERSIONS.contains(version))
        {
            throw new IllegalArgumentException("unsupported header.version " + version);
        }
        this.tensorCount = Math.toIntExact(primitiveReader.readLong(fileChannel));
        this.metadata_kv_count = Math.toIntExact(primitiveReader.readLong(fileChannel));
        this.metadata = HashMap.newHashMap(metadata_kv_count);
        for(int i = 0; i < metadata_kv_count; ++i)
        {
            Pair<String, Object> keyValue = keyValuePairReader.readKeyValuePair(fileChannel);
            assert !metadata.containsKey(keyValue.first());
            metadata.put(keyValue.first(), keyValue.second());
        }
    }


    int getTensorCount()
    {
        return tensorCount;
    }


    Map<String, Object> getMetadata()
    {
        return metadata;
    }
}
