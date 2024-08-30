package io.github.orionlibs.orion_llm4j_llama_inference.models;

import io.github.orionlibs.orion_llm4j_inference.core.model.ModelLoader;
import io.github.orionlibs.orion_llm4j_inference.core.model.Vocabulary;
import io.github.orionlibs.orion_llm4j_inference.core.gguf.GGUFTensorEntry;
import io.github.orionlibs.orion_llm4j_inference.core.gguf.GGUFType;
import io.github.orionlibs.orion_llm4j_inference.core.tensor.FloatTensor;
import io.github.orionlibs.orion_llm4j_llama_inference.core.SimpleLLMProcessor;
import io.github.orionlibs.orion_llm4j_llama_inference.core.SimpleTokenizer;
import io.github.orionlibs.orion_llm4j_llama_inference.core.tensor.Q4_0SimpleFloatTensor;
import io.github.orionlibs.orion_llm4j_llama_inference.core.tensor.Q8_0SimpleFloatTensor;
import io.github.orionlibs.orion_llm4j_llama_inference.core.tensor.SimpleFloatTensor;
import java.io.IOException;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.util.Map;
import java.util.function.IntFunction;

public abstract class SimpleModelLoader extends ModelLoader
{
    protected String TOKENIZER_MODEL;
    protected String PATTERN;


    public SimpleModelLoader(String TOKENIZER_MODEL, String PATTERN)
    {
        this.TOKENIZER_MODEL = TOKENIZER_MODEL;
        this.PATTERN = PATTERN;
    }


    public static SimpleFloatTensor loadQuantized(GGUFTensorEntry entry)
    {
        GGUFType ggmlType = entry.ggmlType();
        return switch(ggmlType)
        {
            //case F32 -> new F32FloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            case Q8_0 -> new Q8_0SimpleFloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            case Q4_0 -> new Q4_0SimpleFloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            default -> throw new UnsupportedOperationException("Quantization format " + ggmlType);
        };
    }


    public static SimpleFloatTensor[] loadArrayOfQuantized(int size, IntFunction<GGUFTensorEntry> getTensorEntry)
    {
        SimpleFloatTensor[] array = new SimpleFloatTensor[size];
        for(int i = 0; i < size; i++)
        {
            array[i] = loadQuantized(getTensorEntry.apply(i));
        }
        return array;
    }


    public static FloatBuffer[] loadArrayOfFloatBuffer(int size, IntFunction<GGUFTensorEntry> getTensorEntry)
    {
        FloatBuffer[] array = new FloatBuffer[size];
        for(int i = 0; i < size; i++)
        {
            array[i] = toFloatBuffer(getTensorEntry.apply(i));
        }
        return array;
    }


    public static FloatBuffer toFloatBuffer(GGUFTensorEntry tensorEntry)
    {
        GGUFType ggmlType = tensorEntry.ggmlType();
        return switch(ggmlType)
        {
            case F32 -> tensorEntry.memorySegment().asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
            default -> throw new UnsupportedOperationException("Conversion to " + ggmlType);
        };
    }


    @Override
    public Vocabulary loadVocabulary(Map<String, Object> metadata)
    {
        String model = (String)metadata.get("tokenizer.ggml.model");
        if(!TOKENIZER_MODEL.equals(model))
        {
            throw new IllegalArgumentException("expected " + TOKENIZER_MODEL + " but found " + model);
        }
        String[] tokens = (String[])metadata.get("tokenizer.ggml.tokens");
        return new Vocabulary(tokens, null);
    }


    public abstract SimpleLLMProcessor loadModel(Path ggufPath, int contextLength) throws IOException;


    protected abstract SimpleTokenizer createTokenizer(Map<String, Object> metadata, Vocabulary vocabulary);
}
