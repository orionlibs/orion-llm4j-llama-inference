package io.github.orionlibs.orion_llm4j_llama_inference.core.sampler;

import io.github.orionlibs.orion_llm4j_llama_inference.core.tensor.SimpleFloatTensor;

@FunctionalInterface
public interface Sampler
{
    int sampleToken(SimpleFloatTensor logits);


    Sampler ARGMAX = SimpleFloatTensor::argmax;
}
