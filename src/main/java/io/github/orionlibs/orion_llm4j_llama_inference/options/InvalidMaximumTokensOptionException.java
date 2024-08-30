package io.github.orionlibs.orion_llm4j_llama_inference.options;

import io.github.orionlibs.orion_assert.OrionUncheckedException;

public class InvalidMaximumTokensOptionException extends OrionUncheckedException
{
    public static final String DefaultErrorMessage = "The given maximum number of tokens is invalid.";


    public InvalidMaximumTokensOptionException()
    {
        super(DefaultErrorMessage);
    }


    public InvalidMaximumTokensOptionException(String message)
    {
        super(message);
    }


    public InvalidMaximumTokensOptionException(String errorMessage, Object... arguments)
    {
        super(String.format(errorMessage, arguments));
    }


    public InvalidMaximumTokensOptionException(Throwable cause, String errorMessage, Object... arguments)
    {
        super(String.format(errorMessage, arguments), cause);
    }


    public InvalidMaximumTokensOptionException(Throwable cause)
    {
        super(cause, DefaultErrorMessage);
    }
}