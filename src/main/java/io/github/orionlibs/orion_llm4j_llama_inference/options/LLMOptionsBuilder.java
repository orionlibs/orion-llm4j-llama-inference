package io.github.orionlibs.orion_llm4j_llama_inference.options;

import io.github.orionlibs.orion_llm4j_inference.config.ConfigurationService;
import io.github.orionlibs.orion_llm4j_inference.options.LLMOptions;

public final class LLMOptionsBuilder
{
    public LLMOptions build()
    {
        LLMOptions options = new LLMOptions();
        options.add("temperature", ConfigurationService.getFloatProp("orion-llm4j-llama-inference.temperature"));
        options.add("randomness", ConfigurationService.getFloatProp("orion-llm4j-llama-inference.randomness"));
        options.add("maximumTokensToProduce", ConfigurationService.getIntegerProp("orion-llm4j-llama-inference.maximum.tokens.to.produce"));
        options.add("interactiveChat", ConfigurationService.getBooleanProp("orion-llm4j-llama-inference.interactive.chat"));
        options.add("asynchronousInference", ConfigurationService.getBooleanProp("orion-llm4j-llama-inference.asynchronous.inference"));
        options.add("llmModelPath", ConfigurationService.getProp("orion-llm4j-llama-inference.llm.model.path"));
        return options;
    }
}