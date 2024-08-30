package io.github.orionlibs.orion_llm4j_llama_inference.core;

import io.github.orionlibs.orion_llm4j_inference.core.Configuration;
import io.github.orionlibs.orion_llm4j_inference.core.LLMProcessor;
import io.github.orionlibs.orion_llm4j_inference.core.Response;
import io.github.orionlibs.orion_llm4j_inference.core.State;
import io.github.orionlibs.orion_llm4j_inference.core.Weights;
import io.github.orionlibs.orion_llm4j_inference.core.sampler.Sampler;
import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;

public abstract class SimpleLLMProcessor extends LLMProcessor
{
    private Response response;


    public SimpleLLMProcessor(Configuration configuration, SimpleTokenizer tokenizer, Weights weights)
    {
        super(configuration, tokenizer, weights, new LlamaNextTokenGenerator());
    }


    public abstract SimpleState createNewState();


    /**
     * LLM generation entry point, ingest prompt tokens and generates new tokens.
     *
     * <p>
     * All prompt tokens are ingested first, then inference starts, until a stop token is found.
     * The returned tokens only include generated/inferred tokens.
     *
     * @param model            model to run inference (including weights, configuration, tokenizer ...)
     * @param state            state of the model e.g. key/value caches ... this is mutated by this call
     * @param startPosition    start prompt ingestion + inference at this position in the context e.g. useful if state was kept across calls (chained generation). 0 implies run with no previous context.
     * @param promptTokens     prompt tokens to ingest, all the prompt tokens will be ingested, given there's enough capacity left in the context
     * @param stopTokens       set of tokens that abort generation during inference, stop tokens do not affect prompt ingestion
     * @param maxTokens        maximum number of tokens (can go up to {@link Configuration#contextLength context length}
     *                         if this value is negative or greater than {@link Configuration#contextLength context length}
     * @param sampler          {@link Sampler strategy} used to select tokens
     * @param onTokenGenerated callback, if non-null, it's called every time a token is inferred e.g. it's not called when ingesting prompt tokens
     * @return Response including the actual model response and list of generated/inferred tokens, including the stop token, if any e.g. does not include any token from the prompt
     */
    @Override
    public Response generateTokens(LLMProcessor model, State state, int startPosition, List<Integer> promptTokens, Set<Integer> stopTokens, int maxTokens, Sampler sampler,
                    IntConsumer onTokenGenerated)
    {
        SimpleState simpleState = (SimpleState)state;
        response = new Response(maxTokens);
        long startNanos = System.nanoTime();
        /*if(maxTokens < 0 || model.getConfiguration().contextLength < maxTokens)
        {
            maxTokens = model.getConfiguration().contextLength;
        }*/
        if(maxTokens < 0)
        {
            maxTokens = model.getConfiguration().contextLength;
        }
        int token = simpleState.latestToken; // BOS?
        int nextToken = 0;
        int promptIndex = 0;
        for(int position = startPosition; position < maxTokens; position++)
        {
            nextTokenGenerator.generate(model, simpleState, token, position);
            //this is the prompt itself
            if(promptIndex < promptTokens.size())
            {
                // Force-pick token from prompt.
                nextToken = promptTokens.get(promptIndex++);
                if(onTokenGenerated != null)
                {
                    onTokenGenerated.accept(nextToken);
                }
                //System.err.print(Tokenizer.replaceControlCharacters(getTokenizer().decode(List.of(nextToken))));
            }
            //this is the LLM response itself
            else
            {
                nextToken = sampler.sampleToken(simpleState.logits);
                response.addResponseToken(nextToken);
                //System.err.print(Tokenizer.replaceControlCharacters(getTokenizer().decode(List.of(nextToken))));
                if(onTokenGenerated != null)
                {
                    onTokenGenerated.accept(nextToken);
                }
                if(stopTokens.contains(nextToken))
                {
                    break;
                }
            }
            simpleState.latestToken = token = nextToken;
        }
        long elapsedNanos = System.nanoTime() - startNanos;
        int numberOfTokensGenerated = promptIndex + response.getResponseTokens().size();
        double tokenGenerationRate = numberOfTokensGenerated / (elapsedNanos / 1_000_000_000.0);
        response.setTokenGenerationRate(tokenGenerationRate);
        response.setNumberOfTokensGenerated(numberOfTokensGenerated);
        response.setStatsFormatted(String.format("%.2f tokens/s (%d)%n", tokenGenerationRate, numberOfTokensGenerated));
        return response;
    }
}
