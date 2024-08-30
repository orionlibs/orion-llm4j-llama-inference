package io.github.orionlibs.orion_llm4j_llama_inference;

import io.github.orionlibs.orion_llm4j_inference.config.ConfigurationService;
import io.github.orionlibs.orion_llm4j_inference.core.inference.ChatFormat;
import io.github.orionlibs.orion_llm4j_inference.core.io.LLMRequest;
import io.github.orionlibs.orion_llm4j_inference.core.io.LLMResponse;
import io.github.orionlibs.orion_llm4j_inference.core.sampler.Sampler;
import io.github.orionlibs.orion_llm4j_inference.options.LLMOptions;
import io.github.orionlibs.orion_llm4j_inference.options.Role;
import io.github.orionlibs.orion_llm4j_llama_inference.core.SimpleTokenGenerationState;
import io.github.orionlibs.orion_llm4j_llama_inference.core.inference.LlamaChatFormat;
import io.github.orionlibs.orion_llm4j_llama_inference.core.sampler.SimpleSamplerSelector;
import io.github.orionlibs.orion_llm4j_llama_inference.model.llama.LlamaModelLoader;
import io.github.orionlibs.orion_llm4j_llama_inference.core.inference.SimpleLlamaProcessor;
import io.github.orionlibs.orion_llm4j_llama_inference.options.InvalidMaximumTokensOptionException;
import io.github.orionlibs.orion_llm4j_llama_inference.options.InvalidUserPromptException;
import io.github.orionlibs.orion_llm4j_llama_inference.options.LLMOptionsBuilder;
import io.github.orionlibs.orion_llm4j_llama_inference.options.MaximumTokenValidator;
import io.github.orionlibs.orion_llm4j_llama_inference.options.UserPromptValidator;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class LLM
{
    private static final String FEATURE_CONFIGURATION_FILE = "/io/github/orionlibs/orion_llm4j_llama_inference/configuration/orion-feature-configuration.prop";
    private LLMOptions options;
    private Sampler sampler;
    private SimpleLlamaProcessor model;
    private boolean isModelLoaded;
    private MaximumTokenValidator maximumTokenValidator;
    private UserPromptValidator userPromptValidator;


    public LLM() throws IOException
    {
        this.maximumTokenValidator = new MaximumTokenValidator();
        this.userPromptValidator = new UserPromptValidator();
        InputStream defaultConfigStream = LLM.class.getResourceAsStream(FEATURE_CONFIGURATION_FILE);
        ConfigurationService.registerConfiguration(defaultConfigStream);
        buildLLMOptions();
        loadModel();
    }


    public LLM(InputStream customConfigStream) throws IOException
    {
        InputStream defaultConfigStream = LLM.class.getResourceAsStream(FEATURE_CONFIGURATION_FILE);
        ConfigurationService.registerConfiguration(defaultConfigStream);
        ConfigurationService.registerConfiguration(customConfigStream);
        buildLLMOptions();
        loadModel();
    }


    public LLMResponse runLLM(String systemPrompt, String userPrompt, int maximumTokensToProduce) throws InvalidMaximumTokensOptionException, InvalidUserPromptException
    {
        maximumTokenValidator.isValidWithException(options, maximumTokensToProduce);
        userPromptValidator.isValidWithException(userPrompt);
        return runPrompt(model, sampler, systemPrompt, userPrompt, maximumTokensToProduce);
    }


    public LLMResponse runLLM(String optionKeyToAdd, Object optionValueToAdd, String systemPrompt, String userPrompt, int maximumTokensToProduce) throws IOException, InvalidMaximumTokensOptionException, InvalidUserPromptException
    {
        options.add(optionKeyToAdd, optionValueToAdd);
        maximumTokenValidator.isValidWithException(options, maximumTokensToProduce);
        userPromptValidator.isValidWithException(userPrompt);
        reloadModelIfOptionsChanged();
        return runPrompt(model, sampler, systemPrompt, userPrompt, maximumTokensToProduce);
    }


    public LLMResponse runLLM(Map<String, Object> optionsToAdd, String systemPrompt, String userPrompt, int maximumTokensToProduce) throws IOException, InvalidMaximumTokensOptionException, InvalidUserPromptException
    {
        options.add(optionsToAdd);
        maximumTokenValidator.isValidWithException(options, maximumTokensToProduce);
        userPromptValidator.isValidWithException(userPrompt);
        reloadModelIfOptionsChanged();
        return runPrompt(model, sampler, systemPrompt, userPrompt, maximumTokensToProduce);
    }


    private void reloadModelIfOptionsChanged() throws IOException
    {
        if(options.haveOptionsChanged)
        {
            isModelLoaded = false;
            loadModel();
        }
    }


    private void buildLLMOptions()
    {
        this.options = new LLMOptionsBuilder().build();
    }


    private void loadModel() throws IOException
    {
        if(!isModelLoaded)
        {
            Path llmModelPath = Paths.get((String)options.getOptionValue("llmModelPath"));
            float temperature = (float)options.getOptionValue("temperature");
            float randomness = (float)options.getOptionValue("randomness");
            model = new LlamaModelLoader().loadModel(llmModelPath, (int)options.getOptionValue("maximumTokensToProduce"));
            sampler = new SimpleSamplerSelector().selectSampler(model.getConfiguration().vocabularySize, temperature, randomness);
            isModelLoaded = true;
        }
    }


    private LLMResponse runPrompt(SimpleLlamaProcessor model, Sampler sampler, String systemPrompt, String userPrompt, int maximumTokensToProduce)
    {
        SimpleTokenGenerationState state = model.createNewState();
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