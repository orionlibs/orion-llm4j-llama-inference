package io.github.orionlibs.orion_llm4j_llama_inference.core.tensor;

import io.github.orionlibs.orion_llm4j_inference.core.gguf.GGUFType;
import java.lang.foreign.MemorySegment;

public record GGUFTensorEntry(MemorySegment mappedFile, String name, GGUFType ggmlType, int[] shape,
                              MemorySegment memorySegment)
{
}
