package io.github.orionlibs.orion_llm4j_llama_inference.options;

import io.github.orionlibs.orion_llm4j_inference.options.LLMOptions;
import io.github.orionlibs.orion_llm4j_inference.options.LLMUserOptionValidator;

public class UserPromptValidator implements LLMUserOptionValidator<String>
{
    @Override
    public boolean isValid(String userPrompt)
    {
        return userPrompt != null && !userPrompt.isEmpty();
    }


    @Override
    public boolean isValid(LLMOptions options, String userPrompt)
    {
        return isValid(userPrompt);
    }


    @Override
    public void isValidWithException(LLMOptions options, String userPrompt) throws InvalidUserPromptException
    {
        isValidWithException(userPrompt);
    }


    @Override
    public void isValidWithException(String userPrompt) throws InvalidUserPromptException
    {
        if(!isValid(userPrompt))
        {
            throw new InvalidUserPromptException(InvalidUserPromptException.DefaultErrorMessage);
        }
    }
}