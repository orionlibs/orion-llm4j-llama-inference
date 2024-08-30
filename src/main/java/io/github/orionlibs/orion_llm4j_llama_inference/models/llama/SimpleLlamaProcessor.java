package io.github.orionlibs.orion_llm4j_llama_inference.models.llama;

import io.github.orionlibs.orion_llm4j_inference.core.inference.LLMConfiguration;
import io.github.orionlibs.orion_llm4j_inference.core.model.Weights;
import io.github.orionlibs.orion_llm4j_llama_inference.core.SimpleLLMProcessor;
import io.github.orionlibs.orion_llm4j_llama_inference.core.SimpleTokenGenerationState;
import io.github.orionlibs.orion_llm4j_llama_inference.core.SimpleTokenizer;

public final class SimpleLlamaProcessor extends SimpleLLMProcessor
{
    public SimpleLlamaProcessor(LLMConfiguration configuration, SimpleTokenizer tokenizer, Weights weights)
    {
        super(configuration, tokenizer, weights);
    }


    public SimpleTokenGenerationState createNewState()
    {
        SimpleTokenGenerationState state = new SimpleTokenGenerationState(getConfiguration());
        state.latestToken = getTokenizer().getSpecialTokens().get("<|begin_of_text|>");
        return state;
    }


    @Override
    public SimpleTokenizer getTokenizer()
    {
        return (SimpleTokenizer)super.getTokenizer();
    }
}
