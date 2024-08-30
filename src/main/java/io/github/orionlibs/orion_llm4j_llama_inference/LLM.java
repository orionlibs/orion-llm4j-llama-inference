package io.github.orionlibs.orion_llm4j_llama_inference;

import io.github.orionlibs.orion_llm4j_inference.config.ConfigurationService;
import io.github.orionlibs.orion_llm4j_inference.core.ChatFormat;
import io.github.orionlibs.orion_llm4j_inference.core.Message;
import io.github.orionlibs.orion_llm4j_inference.core.Response;
import io.github.orionlibs.orion_llm4j_inference.core.Role;
import io.github.orionlibs.orion_llm4j_inference.core.sampler.Sampler;
import io.github.orionlibs.orion_llm4j_inference.options.LLMOptions;
import io.github.orionlibs.orion_llm4j_llama_inference.core.SimpleState;
import io.github.orionlibs.orion_llm4j_llama_inference.core.sampler.SimpleSamplerSelector;
import io.github.orionlibs.orion_llm4j_llama_inference.models.llama.LlamaChatFormat;
import io.github.orionlibs.orion_llm4j_llama_inference.models.llama.LlamaSimpleModelLoader;
import io.github.orionlibs.orion_llm4j_llama_inference.models.llama.SimpleLlamaProcessor;
import io.github.orionlibs.orion_llm4j_llama_inference.options.LLMOptionsBuilder;
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


    public LLM() throws IOException
    {
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


    public Response runLLM(String systemPrompt, String userPrompt, int maximumTokensToProduce)
    {
        return runPrompt(model, sampler, options, systemPrompt, userPrompt, maximumTokensToProduce);
    }


    public Response runLLM(String optionKeyToAdd, Object optionValueToAdd, String systemPrompt, String userPrompt, int maximumTokensToProduce) throws IOException
    {
        options.add(optionKeyToAdd, optionValueToAdd);
        reloadModelIfOptionsChanged();
        return runPrompt(model, sampler, options, systemPrompt, userPrompt, maximumTokensToProduce);
    }


    public Response runLLM(Map<String, Object> optionsToAdd, String systemPrompt, String userPrompt, int maximumTokensToProduce) throws IOException
    {
        options.add(optionsToAdd);
        reloadModelIfOptionsChanged();
        return runPrompt(model, sampler, options, systemPrompt, userPrompt, maximumTokensToProduce);
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
            model = new LlamaSimpleModelLoader().loadModel(llmModelPath, (int)options.getOptionValue("maximumTokensToProduce"));
            sampler = new SimpleSamplerSelector().selectSampler(model.getConfiguration().vocabularySize, temperature, randomness);
            isModelLoaded = true;
        }
    }


    private Response runPrompt(SimpleLlamaProcessor model, Sampler sampler, LLMOptions options, String systemPrompt, String userPrompt, int maximumTokensToProduce)
    {
        SimpleState state = model.createNewState();
        ChatFormat chatFormat = new LlamaChatFormat(model.getTokenizer());
        List<Integer> promptTokens = new ArrayList<>();
        promptTokens.add(chatFormat.getBeginOfText());
        if(systemPrompt != null)
        {
            promptTokens.addAll(chatFormat.encodeMessage(new Message(Role.SYSTEM, systemPrompt)));
        }
        promptTokens.addAll(chatFormat.encodeMessage(new Message(Role.USER, userPrompt)));
        promptTokens.addAll(chatFormat.encodeHeader(new Message(Role.ASSISTANT, "")));
        Set<Integer> stopTokens = chatFormat.getStopTokens();
        Response response = model.generateTokens(model, state, 0, promptTokens, stopTokens, maximumTokensToProduce, sampler, token -> {
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