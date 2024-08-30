package io.github.orionlibs.orion_llm4j_llama_inference.options;

import io.github.orionlibs.orion_llm4j_inference.options.LLMOptions;
import io.github.orionlibs.orion_llm4j_inference.options.LLMUserOptionValidator;

public class MaximumTokenValidator implements LLMUserOptionValidator<Integer>
{
    private int maximumTokensAllowedToProduce;


    @Override
    public boolean isValid(Integer maximumTokensToProduceToCheck)
    {
        throw new UnsupportedOperationException();
    }


    @Override
    public boolean isValid(LLMOptions options, Integer maximumTokensToProduceToCheck)
    {
        maximumTokensAllowedToProduce = (int)options.getOptionValue("maximumTokensToProduce");
        return maximumTokensToProduceToCheck.intValue() <= maximumTokensAllowedToProduce;
    }


    @Override
    public void isValidWithException(LLMOptions options, Integer maximumTokensToProduceToCheck) throws InvalidMaximumTokensOptionException
    {
        if(!isValid(options, maximumTokensToProduceToCheck))
        {
            throw new InvalidMaximumTokensOptionException(STR."\{InvalidMaximumTokensOptionException.DefaultErrorMessage} Expected up to \{maximumTokensAllowedToProduce}, but got \{Integer.toString(maximumTokensToProduceToCheck)}");
        }
    }


    @Override
    public void isValidWithException(Integer maximumTokensToProduceToCheck)
    {
        throw new UnsupportedOperationException();
    }
}