package io.github.orionlibs.orion_llm4j_llama_inference.core;

import java.util.List;
import java.util.Set;

public abstract class ChatFormat
{
    protected final SimpleTokenizer tokenizer;
    protected int beginOfText;


    public ChatFormat(SimpleTokenizer tokenizer)
    {
        this.tokenizer = tokenizer;
    }


    public abstract List<Integer> encodeMessage(Message message);


    public abstract List<Integer> encodeHeader(Message message);


    public abstract Set<Integer> getStopTokens();


    public SimpleTokenizer getTokenizer()
    {
        return tokenizer;
    }


    public int getBeginOfText()
    {
        return beginOfText;
    }
}
