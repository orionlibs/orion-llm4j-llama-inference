package io.github.orionlibs.orion_llm4j_llama_inference.core.sampler;

import io.github.orionlibs.orion_llm4j_inference.core.sampler.Sampler;
import io.github.orionlibs.orion_llm4j_inference.core.sampler.SamplerSelector;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

public class SimpleSamplerSelector implements SamplerSelector
{
    @Override
    public Sampler selectSampler(int vocabularySize, float temperature, float topp)
    {
        Sampler sampler;
        if(temperature == 0.0f)
        {
            // greedy argmax sampling: take the token with the highest probability
            sampler = Sampler.ARGMAX;
        }
        else
        {
            // we sample from this distribution to get the next token
            RandomGenerator rng = RandomGeneratorFactory.getDefault().create(System.nanoTime());
            Sampler innerSampler;
            if(topp <= 0 || topp >= 1)
            {
                // simply sample from the predicted probability distribution
                innerSampler = new CategoricalSampler(rng);
            }
            else
            {
                // top-p (nucleus) sampling, clamping the least likely tokens to zero
                innerSampler = new SimpleToppSampler(vocabularySize, topp, rng);
            }
            sampler = logits -> {
                // apply the temperature to the logits
                logits.divideInPlace(0, logits.size(), temperature);
                // apply softmax to the logits to get the probabilities for next token
                logits.softmaxInPlace(0, logits.size());
                return innerSampler.sampleToken(logits);
            };
        }
        return sampler;
    }
}