package io.github.orionlibs.orion_llm4j_llama_inference.core;

import io.github.orionlibs.orion_llm4j_llama_inference.core.tensor.SimpleFloatTensor;
import java.nio.FloatBuffer;

public final class Weights
{
    // token embedding table
    public final SimpleFloatTensor token_embedding_table; // (vocab_size, dim)
    // weights for rmsnorms
    public final FloatBuffer[] rms_att_weight; // (layer, dim) rmsnorm weights
    // weights for matmuls
    public final SimpleFloatTensor[] wq; // (layer, n_heads * head_size)
    public final SimpleFloatTensor[] wk; // (layer, n_kv_heads, head_size)
    public final SimpleFloatTensor[] wv; // (layer, n_kv_heads * head_size)
    public final SimpleFloatTensor[] wo; // (layer, n_heads * head_size, dim)
    public final FloatBuffer[] rms_ffn_weight; // (layer, dim)
    // weights for ffn
    public final SimpleFloatTensor[] w1; // (layer, hidden_dim, dim)
    public final SimpleFloatTensor[] w2; // (layer, dim, hidden_dim)
    public final SimpleFloatTensor[] w3; // (layer, hidden_dim, dim)
    // public final rmsnorm
    public final FloatBuffer rms_final_weight; // (dim,)
    // freq_cis for RoPE relatively positional embeddings
    public final FloatBuffer freq_cis_real; // (seq_len, head_size/2)
    public final FloatBuffer freq_cis_imag; // (seq_len, head_size/2)
    // (optional) classifier weights for the logits, on the last layer
    public final SimpleFloatTensor wcls; // (vocab_size, dim)


    public Weights(SimpleFloatTensor token_embedding_table, FloatBuffer[] rms_att_weight, SimpleFloatTensor[] wq, SimpleFloatTensor[] wk, SimpleFloatTensor[] wv, SimpleFloatTensor[] wo, FloatBuffer[] rms_ffn_weight, SimpleFloatTensor[] w1, SimpleFloatTensor[] w2, SimpleFloatTensor[] w3, FloatBuffer rms_final_weight, FloatBuffer freq_cis_real,
                    FloatBuffer freq_cis_imag, SimpleFloatTensor wcls)
    {
        this.token_embedding_table = token_embedding_table;
        this.rms_att_weight = rms_att_weight;
        this.wq = wq;
        this.wk = wk;
        this.wv = wv;
        this.wo = wo;
        this.rms_ffn_weight = rms_ffn_weight;
        this.w1 = w1;
        this.w2 = w2;
        this.w3 = w3;
        this.rms_final_weight = rms_final_weight;
        this.freq_cis_real = freq_cis_real;
        this.freq_cis_imag = freq_cis_imag;
        this.wcls = wcls;
    }
}
