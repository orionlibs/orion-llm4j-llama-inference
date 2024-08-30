package io.github.orionlibs.orion_llm4j_llama_inference.model;

import io.github.orionlibs.orion_llm4j_inference.core.model.ModelLoader;
import io.github.orionlibs.orion_llm4j_inference.core.model.Vocabulary;
import io.github.orionlibs.orion_llm4j_llama_inference.core.SimpleLLMProcessor;
import io.github.orionlibs.orion_llm4j_llama_inference.core.SimpleTokenizer;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;

public abstract class AbstractModelLoader extends ModelLoader
{
    protected String TOKENIZER_MODEL;
    protected String PATTERN;


    public AbstractModelLoader(String TOKENIZER_MODEL, String PATTERN)
    {
        this.TOKENIZER_MODEL = TOKENIZER_MODEL;
        this.PATTERN = PATTERN;
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
