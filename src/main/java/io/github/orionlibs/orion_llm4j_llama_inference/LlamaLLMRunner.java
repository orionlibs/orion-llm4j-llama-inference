package io.github.orionlibs.orion_llm4j_llama_inference;

import io.github.orionlibs.orion_llm4j_inference.core.inference.ChatFormat;
import io.github.orionlibs.orion_llm4j_inference.core.io.LLMRequest;
import io.github.orionlibs.orion_llm4j_inference.core.io.LLMResponse;
import io.github.orionlibs.orion_llm4j_inference.core.sampler.Sampler;
import io.github.orionlibs.orion_llm4j_inference.options.Role;
import io.github.orionlibs.orion_llm4j_llama_inference.core.LlamaTokenGenerationState;
import io.github.orionlibs.orion_llm4j_llama_inference.core.inference.LlamaChatFormat;
import io.github.orionlibs.orion_llm4j_llama_inference.core.inference.LlamaLLMInferencer;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

record LlamaLLMRunner(LlamaLLMInferencer model, Sampler sampler, String systemPrompt, String userPrompt, int maximumTokensToProduce)
{
    LLMResponse runPrompt()
    {
        LlamaTokenGenerationState state = model.createNewState();
        ChatFormat chatFormat = new LlamaChatFormat(model.getTokenizer());
        List<Integer> promptTokens = new ArrayList<>();
        promptTokens.add(chatFormat.getBeginOfText());
        if(systemPrompt != null)
        {
            promptTokens.addAll(chatFormat.encodeMessage(new LLMRequest(Role.SYSTEM, systemPrompt)));
        }
        promptTokens.addAll(chatFormat.encodeMessage(new LLMRequest(Role.USER, userPrompt)));
        promptTokens.addAll(chatFormat.encodeHeader(new LLMRequest(Role.ASSISTANT, "")));
        Set<Integer> stopTokens = chatFormat.getStopTokens();
        LLMResponse response = model.generateTokens(state, 0, promptTokens, stopTokens, maximumTokensToProduce, sampler, token -> {
            if(!model.getTokenizer().isSpecialToken(token))
            {
                System.out.print(model.getTokenizer().decode(List.of(token)));
            }
        });
        if(!response.getResponseTokens().isEmpty() && stopTokens.contains(response.getResponseTokens().getLast()))
        {
            response.getResponseTokens().removeLast();
        }
        String responseText = model.getTokenizer().decode(response.getResponseTokens());
        response.appendContent(responseText);
        return response;
    }
}