package io.github.orionlibs.orion_llm4j_llama_inference.options;

import io.github.orionlibs.orion_assert.OrionUncheckedException;

public class InvalidUserPromptException extends OrionUncheckedException
{
    public static final String DefaultErrorMessage = "The given user prompt is invalid.";


    public InvalidUserPromptException()
    {
        super(DefaultErrorMessage);
    }


    public InvalidUserPromptException(String message)
    {
        super(message);
    }


    public InvalidUserPromptException(String errorMessage, Object... arguments)
    {
        super(String.format(errorMessage, arguments));
    }


    public InvalidUserPromptException(Throwable cause, String errorMessage, Object... arguments)
    {
        super(String.format(errorMessage, arguments), cause);
    }


    public InvalidUserPromptException(Throwable cause)
    {
        super(cause, DefaultErrorMessage);
    }
}