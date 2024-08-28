package io.github.orionlibs.orion_llm4j_llama_inference.core.gguf;

import io.github.orionlibs.orion_llm4j_inference.core.utils.Pair;
import java.io.IOException;
import java.nio.channels.FileChannel;

final class GGUFModelKeyValuePairReader
{
    private GGUFModelStringReader stringReader;
    private GGUFModelMetadataReader metadataReader;


    GGUFModelKeyValuePairReader()
    {
        this.stringReader = new GGUFModelStringReader();
        this.metadataReader = new GGUFModelMetadataReader();
    }


    Pair<String, Object> readKeyValuePair(FileChannel fileChannel) throws IOException
    {
        String key = stringReader.readString(fileChannel);
        assert key.length() < (1 << 16);
        assert key.codePoints().allMatch(cp -> ('a' <= cp && cp <= 'z') || ('0' <= cp && cp <= '9') || cp == '_' || cp == '.');
        Object value = metadataReader.readMetadataValue(fileChannel);
        return new Pair<>(key, value);
    }
}
