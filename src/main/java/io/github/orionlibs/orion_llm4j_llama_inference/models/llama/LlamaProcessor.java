package io.github.orionlibs.orion_llm4j_llama_inference.models.llama;

import io.github.orionlibs.orion_llm4j_llama_inference.core.Configuration;
import io.github.orionlibs.orion_llm4j_llama_inference.core.LLMProcessor;
import io.github.orionlibs.orion_llm4j_llama_inference.core.State;
import io.github.orionlibs.orion_llm4j_llama_inference.core.Tokenizer;
import io.github.orionlibs.orion_llm4j_llama_inference.core.Weights;

public final class LlamaProcessor extends LLMProcessor
{
    public LlamaProcessor(Configuration configuration, Tokenizer tokenizer, Weights weights)
    {
        super(configuration, tokenizer, weights);
    }


    @Override
    public State createNewState()
    {
        State state = new State(configuration);
        state.latestToken = tokenizer.getSpecialTokens().get("<|begin_of_text|>");
        return state;
    }


    public Configuration getConfiguration()
    {
        return configuration;
    }


    public Tokenizer getTokenizer()
    {
        return tokenizer;
    }
}
