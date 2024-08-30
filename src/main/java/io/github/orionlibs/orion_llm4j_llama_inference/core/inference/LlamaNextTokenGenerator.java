package io.github.orionlibs.orion_llm4j_llama_inference.core.inference;

import io.github.orionlibs.orion_llm4j_inference.core.inference.LLMConfiguration;
import io.github.orionlibs.orion_llm4j_inference.core.inference.LLMInferencer;
import io.github.orionlibs.orion_llm4j_inference.core.inference.NextTokenGenerator;
import io.github.orionlibs.orion_llm4j_inference.core.model.Weights;
import io.github.orionlibs.orion_llm4j_inference.core.token.TokenGenerationState;
import io.github.orionlibs.orion_llm4j_inference.core.utils.Parallel;
import io.github.orionlibs.orion_llm4j_llama_inference.core.LlamaTokenGenerationState;
import io.github.orionlibs.orion_llm4j_llama_inference.core.tensor.SimpleFloatTensor;
import java.nio.FloatBuffer;

public class LlamaNextTokenGenerator implements NextTokenGenerator
{
    private void rmsnorm(SimpleFloatTensor out, SimpleFloatTensor x, FloatBuffer weight, int size, float rmsNormEps)
    {
        // calculate sum of squares
        float ss = x.reduce(0, size, 0f, (acc, xi) -> acc + xi * xi);
        ss /= size;
        ss += rmsNormEps;
        ss = (float)(1.0 / Math.sqrt(ss));
        // normalize and scale
        final float finalss = ss; // for the lambda
        out.mapWithIndexInPlace(0, size, (value, index) -> weight.get(index) * (finalss * x.getFloat(index)));
    }


    @Override
    public SimpleFloatTensor generate(LLMInferencer model, TokenGenerationState state, int token, int position)
    {
        LlamaTokenGenerationState simpleState = (LlamaTokenGenerationState)state;
        // a few convenience variables
        LLMConfiguration config = model.getConfiguration();
        Weights weights = model.getWeights();
        int dim = config.dim;
        int headSize = config.headSize;
        int kvDim = (config.dim * config.numberOfKeyValueHeads) / config.numberOfHeads;
        int kvMul = config.numberOfHeads / config.numberOfKeyValueHeads; // integer multiplier of the kv sharing in multiquery
        float sqrtHeadSize = (float)Math.sqrt(headSize);
        // copy the token embedding into x
        weights.token_embedding_table.copyTo(token * dim, simpleState.x, 0, dim);
        // forward all the layers
        for(int l = 0; l < config.numberOfLayers; l++)
        {
            // attention rmsnorm
            rmsnorm(simpleState.xb, simpleState.x, weights.rms_att_weight[l], dim, config.rmsNormEps);
            // qkv matmuls for this position
            weights.wq[l].matmul(simpleState.xb, simpleState.q, dim, dim);
            weights.wk[l].matmul(simpleState.xb, simpleState.k, kvDim, dim);
            weights.wv[l].matmul(simpleState.xb, simpleState.v, kvDim, dim);
            // RoPE relative positional encoding: complex-valued rotate q and k in each head
            for(int i = 0; i < dim; i += 2)
            {
                int head_dim = i % headSize;
                float fcr = weights.freq_cis_real.get(position * (headSize / 2) + (head_dim / 2));
                float fci = weights.freq_cis_imag.get(position * (headSize / 2) + (head_dim / 2));
                int rotn = i < kvDim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
                for(int v = 0; v < rotn; v++)
                {
                    SimpleFloatTensor vec = v == 0 ? simpleState.q : simpleState.k; // the vector to rotate (query or key)
                    float v0 = vec.getFloat(i);
                    float v1 = vec.getFloat(i + 1);
                    vec.setFloat(i, v0 * fcr - v1 * fci);
                    vec.setFloat(i + 1, v0 * fci + v1 * fcr);
                }
            }
            // save key,value at this time step (position) to our kv cache
            //int loff = l * config.seq_len * kvDim; // kv cache layer offset for convenience
            simpleState.k.copyTo(0, simpleState.keyCache[l], position * kvDim, kvDim);
            simpleState.v.copyTo(0, simpleState.valueCache[l], position * kvDim, kvDim);
            int curLayer = l;
            // multihead attention. iterate over all heads
            Parallel.parallelFor(0, config.numberOfHeads, h -> {
                // get the query vector for this head
                // float* q = s.q + h * headSize;
                int qOffset = h * headSize;
                // attention scores for this head
                // float* att = s.att + h * config.seq_len;
                int attOffset = h * config.contextLength;
                // iterate over all timesteps, including the current one
                for(int t = 0; t <= position; t++)
                {
                    // get the key vector for this head and at this timestep
                    // float* k = s.key_cache + loff + t * dim + h * headSize;
                    int keyCacheOffset = /* loff + */ t * kvDim + (h / kvMul) * headSize;
                    // calculate the attention score as the dot product of q and k
                    float score = simpleState.q.dot(qOffset, simpleState.keyCache[curLayer], keyCacheOffset, headSize);
                    score /= sqrtHeadSize;
                    // save the score to the attention buffer
                    simpleState.att.setFloat(attOffset + t, score);
                }
                // softmax the scores to get attention weights, from 0..position inclusively
                simpleState.att.softmaxInPlace(attOffset, position + 1);
                // weighted sum of the values, store back into xb
                // float* xb = s.xb + h * headSize;
                int xbOffset = h * headSize;
                // memset(xb, 0, headSize * sizeof(float));
                simpleState.xb.fillInPlace(xbOffset, headSize, 0f);
                for(int t = 0; t <= position; t++)
                {
                    // get the value vector for this head and at this timestep
                    // float* v = s.value_cache + loff + t * dim + h * headSize;
                    int vOffset = /* loff + */ t * kvDim + (h / kvMul) * headSize;
                    // get the attention weight for this timestep
                    float a = simpleState.att.getFloat(attOffset + t);
                    // accumulate the weighted value into xb
                    simpleState.xb.saxpyInPlace(xbOffset, simpleState.valueCache[curLayer], vOffset, headSize, a);
                }
            });
            // final matmul to get the output of the attention
            weights.wo[l].matmul(simpleState.xb, simpleState.xb2, dim, dim);
            // residual connection back into x
            simpleState.x.addInPlace(simpleState.xb2);
            // ffn rmsnorm
            rmsnorm(simpleState.xb, simpleState.x, weights.rms_ffn_weight[l], dim, config.rmsNormEps);
            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // first calculate self.w1(x) and self.w3(x)
            weights.w1[l].matmul(simpleState.xb, simpleState.hb, config.hiddenDim, dim);
            weights.w3[l].matmul(simpleState.xb, simpleState.hb2, config.hiddenDim, dim);
            // SwiGLU non-linearity
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            simpleState.hb.mapInPlace(value -> value / (float)(1.0 + Math.exp(-value)));
            // elementwise multiply with w3(x)
            simpleState.hb.multiplyInPlace(simpleState.hb2);
            // final matmul to get the output of the ffn
            weights.w2[l].matmul(simpleState.hb, simpleState.xb, dim, config.hiddenDim);
            // residual connection
            simpleState.x.addInPlace(simpleState.xb);
        }
        // final rmsnorm
        rmsnorm(simpleState.x, simpleState.x, weights.rms_final_weight, dim, config.rmsNormEps);
        // classifier into logits
        weights.wcls.matmul(simpleState.x, simpleState.logits, config.vocabularySize, dim);
        return simpleState.logits;
    }
}
