package io.github.orionlibs.orion_llm4j_llama_inference.options;

import io.github.orionlibs.orion_llm4j_inference.options.LLMOptions;
import io.github.orionlibs.orion_llm4j_inference.config.ConfigurationService;

public final class LLMOptionsBuilder
{
    public LLMOptions build()
    {
        LLMOptions options = new LLMOptions();
        options.add("temperature", ConfigurationService.getFloatProp("orion-llm4j-llama-inference.temperature"));
        options.add("randomness", ConfigurationService.getFloatProp("orion-llm4j-llama-inference.randomness"));
        options.add("maximumTokensToProduce", ConfigurationService.getIntegerProp("orion-llm4j-llama-inference.maximum.tokens.to.produce"));
        options.add("interactiveChat", ConfigurationService.getBooleanProp("orion-llm4j-llama-inference.interactive.chat"));
        options.add("llmModelPath", ConfigurationService.getProp("orion-llm4j-llama-inference.llm.model.path"));
        return options;
    }
}