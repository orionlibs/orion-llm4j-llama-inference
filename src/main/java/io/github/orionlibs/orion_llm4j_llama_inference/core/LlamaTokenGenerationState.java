package io.github.orionlibs.orion_llm4j_llama_inference.core;

import io.github.orionlibs.orion_llm4j_inference.core.inference.LLMConfiguration;
import io.github.orionlibs.orion_llm4j_inference.core.token.TokenGenerationState;
import io.github.orionlibs.orion_llm4j_llama_inference.core.tensor.ArraySimpleFloatTensor;
import io.github.orionlibs.orion_llm4j_llama_inference.core.tensor.SimpleFloatTensor;
import java.util.stream.Stream;

public final class LlamaTokenGenerationState extends TokenGenerationState
{
    // current wave of activations
    public final SimpleFloatTensor x; // activation at current time stamp (dim,)
    public final SimpleFloatTensor xb; // same, but inside a residual branch (dim,)
    public final SimpleFloatTensor xb2; // an additional buffer just for convenience (dim,)
    public final SimpleFloatTensor hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    public final SimpleFloatTensor hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    public final SimpleFloatTensor q; // query (dim,)
    public final SimpleFloatTensor k; // key (dim,)
    public final SimpleFloatTensor v; // value (dim,)
    public final SimpleFloatTensor att; // buffer for scores/attention values (n_heads, seq_len)
    public final SimpleFloatTensor logits; // output logits
    // kv cache
    public final SimpleFloatTensor[] keyCache;   // (n_layer, seq_len, kv_dim)
    public final SimpleFloatTensor[] valueCache; // (n_layer, seq_len, kv_dim)


    public LlamaTokenGenerationState(LLMConfiguration config)
    {
        this.x = ArraySimpleFloatTensor.allocate(config.dim);
        this.xb = ArraySimpleFloatTensor.allocate(config.dim);
        this.xb2 = ArraySimpleFloatTensor.allocate(config.dim);
        this.hb = ArraySimpleFloatTensor.allocate(config.hiddenDim);
        this.hb2 = ArraySimpleFloatTensor.allocate(config.hiddenDim);
        this.q = ArraySimpleFloatTensor.allocate(config.dim);
        this.k = ArraySimpleFloatTensor.allocate(config.dim);
        this.v = ArraySimpleFloatTensor.allocate(config.dim);
        this.att = ArraySimpleFloatTensor.allocate(config.numberOfHeads, config.contextLength);
        this.logits = ArraySimpleFloatTensor.allocate(config.vocabularySize);
        int kvDim = (config.dim * config.numberOfKeyValueHeads) / config.numberOfHeads;
        this.keyCache = Stream.generate(() -> ArraySimpleFloatTensor.allocate(config.contextLength, kvDim)).limit(config.numberOfLayers).toArray(SimpleFloatTensor[]::new);
        this.valueCache = Stream.generate(() -> ArraySimpleFloatTensor.allocate(config.contextLength, kvDim)).limit(config.numberOfLayers).toArray(SimpleFloatTensor[]::new);
    }
}
