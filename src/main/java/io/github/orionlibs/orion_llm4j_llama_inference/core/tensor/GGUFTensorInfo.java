package io.github.orionlibs.orion_llm4j_llama_inference.core.tensor;

import io.github.orionlibs.orion_llm4j_inference.core.gguf.GGUFType;

public record GGUFTensorInfo(String name, int[] dimensions, GGUFType ggmlType, long offset)
{
}
