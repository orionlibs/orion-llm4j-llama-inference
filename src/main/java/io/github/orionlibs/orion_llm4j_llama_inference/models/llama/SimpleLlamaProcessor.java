package io.github.orionlibs.orion_llm4j_llama_inference.models.llama;

import io.github.orionlibs.orion_llm4j_inference.core.Configuration;
import io.github.orionlibs.orion_llm4j_inference.core.Weights;
import io.github.orionlibs.orion_llm4j_llama_inference.core.SimpleLLMProcessor;
import io.github.orionlibs.orion_llm4j_llama_inference.core.SimpleState;
import io.github.orionlibs.orion_llm4j_llama_inference.core.SimpleTokenizer;

public final class SimpleLlamaProcessor extends SimpleLLMProcessor
{
    public SimpleLlamaProcessor(Configuration configuration, SimpleTokenizer tokenizer, Weights weights)
    {
        super(configuration, tokenizer, weights);
    }


    @Override
    public SimpleState createNewState()
    {
        SimpleState state = new SimpleState(configuration);
        state.latestToken = tokenizer.getSpecialTokens().get("<|begin_of_text|>");
        return state;
    }


    public Configuration getConfiguration()
    {
        return configuration;
    }


    public SimpleTokenizer getTokenizer()
    {
        return (SimpleTokenizer)tokenizer;
    }
}
