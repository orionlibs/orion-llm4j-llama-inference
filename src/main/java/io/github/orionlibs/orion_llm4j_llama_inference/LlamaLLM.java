package io.github.orionlibs.orion_llm4j_llama_inference;

import io.github.orionlibs.orion_llm4j_inference.config.ConfigurationService;
import io.github.orionlibs.orion_llm4j_inference.core.io.LLMResponse;
import io.github.orionlibs.orion_llm4j_inference.core.sampler.Sampler;
import io.github.orionlibs.orion_llm4j_inference.options.LLMOptions;
import io.github.orionlibs.orion_llm4j_llama_inference.core.inference.LlamaLLMInferencer;
import io.github.orionlibs.orion_llm4j_llama_inference.core.sampler.SimpleSamplerSelector;
import io.github.orionlibs.orion_llm4j_llama_inference.model.LlamaModelLoader;
import io.github.orionlibs.orion_llm4j_llama_inference.options.InvalidMaximumTokensOptionException;
import io.github.orionlibs.orion_llm4j_llama_inference.options.InvalidUserPromptException;
import io.github.orionlibs.orion_llm4j_llama_inference.options.LLMOptionsBuilder;
import io.github.orionlibs.orion_llm4j_llama_inference.options.MaximumTokenValidator;
import io.github.orionlibs.orion_llm4j_llama_inference.options.UserPromptValidator;
import io.github.orionlibs.orion_task_runner.OrionJobService;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import java.util.concurrent.ExecutionException;

public class LlamaLLM
{
    private static final String FEATURE_CONFIGURATION_FILE = "/io/github/orionlibs/orion_llm4j_llama_inference/configuration/orion-feature-configuration.prop";
    private LLMOptions options;
    private Sampler sampler;
    private LlamaLLMInferencer model;
    private boolean isModelLoaded;
    private MaximumTokenValidator maximumTokenValidator;
    private UserPromptValidator userPromptValidator;


    public LlamaLLM() throws IOException
    {
        this.maximumTokenValidator = new MaximumTokenValidator();
        this.userPromptValidator = new UserPromptValidator();
        InputStream defaultConfigStream = LlamaLLM.class.getResourceAsStream(FEATURE_CONFIGURATION_FILE);
        ConfigurationService.registerConfiguration(defaultConfigStream);
        buildLLMOptions();
    }


    public LlamaLLM(InputStream customConfigStream) throws IOException
    {
        InputStream defaultConfigStream = LlamaLLM.class.getResourceAsStream(FEATURE_CONFIGURATION_FILE);
        ConfigurationService.registerConfiguration(defaultConfigStream);
        ConfigurationService.registerConfiguration(customConfigStream);
        buildLLMOptions();
    }


    public LLMResponse runLLM(String systemPrompt, String userPrompt, int maximumTokensToProduce) throws InvalidMaximumTokensOptionException, InvalidUserPromptException, IOException
    {
        loadModel();
        maximumTokenValidator.isValidWithException(options, maximumTokensToProduce);
        userPromptValidator.isValidWithException(userPrompt);
        return runPrompt(model, sampler, systemPrompt, userPrompt, maximumTokensToProduce);
    }


    public LLMResponse runLLM(String optionKeyToAdd, Object optionValueToAdd, String systemPrompt, String userPrompt, int maximumTokensToProduce) throws IOException, InvalidMaximumTokensOptionException, InvalidUserPromptException
    {
        loadModel();
        options.add(optionKeyToAdd, optionValueToAdd);
        maximumTokenValidator.isValidWithException(options, maximumTokensToProduce);
        userPromptValidator.isValidWithException(userPrompt);
        reloadModelIfOptionsChanged();
        return runPrompt(model, sampler, systemPrompt, userPrompt, maximumTokensToProduce);
    }


    public LLMResponse runLLM(Map<String, Object> optionsToAdd, String systemPrompt, String userPrompt, int maximumTokensToProduce) throws IOException, InvalidMaximumTokensOptionException, InvalidUserPromptException
    {
        loadModel();
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


    private LLMResponse runPrompt(LlamaLLMInferencer model, Sampler sampler, String systemPrompt, String userPrompt, int maximumTokensToProduce)
    {
        LlamaLLMRunner runner = new LlamaLLMRunner(model, sampler, systemPrompt, userPrompt, maximumTokensToProduce);
        try
        {
            return new OrionJobService<LLMResponse>().runJobAndGetResult(() -> runner.runPrompt());
        }
        catch(ExecutionException e)
        {
            throw new RuntimeException(e);
        }
        catch(InterruptedException e)
        {
            throw new RuntimeException(e);
        }
    }
}