package io.github.orionlibs.orion_llm4j_llama_inference.core.utils;

@FunctionalInterface
public interface MapWithIndexFunction
{
    float apply(float value, int index);
}
