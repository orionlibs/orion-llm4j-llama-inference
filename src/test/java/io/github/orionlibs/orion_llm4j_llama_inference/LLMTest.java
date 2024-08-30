package io.github.orionlibs.orion_llm4j_llama_inference;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.orionlibs.orion_llm4j_inference.core.io.LLMResponse;
import io.github.orionlibs.orion_llm4j_llama_inference.options.InvalidMaximumTokensOptionException;
import io.github.orionlibs.orion_llm4j_llama_inference.options.InvalidUserPromptException;
import java.io.IOException;
import java.io.InputStream;
import org.apache.commons.io.IOUtils;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.TestInstance.Lifecycle;

//@ActiveProfiles("testing")
@TestInstance(Lifecycle.PER_CLASS)
//@Execution(ExecutionMode.CONCURRENT)
//@RunWith(JUnitPlatform.class)
public class LLMTest
{
    //@Autowired
    private LLM llm;


    @BeforeEach
    void setUp() throws IOException
    {
        llm = new LLM();
    }


    @Test
    void test_main()
    {
        LLMResponse response = llm.runLLM("Answer in no more than 12 words. Start your answer with the words \"The sky appears blue, because\"", "Why is the sky blue?", 512);
        String capturedOutput = response.getContent();
        System.out.println(capturedOutput);
        assertTrue(capturedOutput.startsWith("The sky appears blue, because"));
    }


    @Test
    @Disabled
    void test_book() throws IOException
    {
        InputStream defaultConfigStream = LLM.class.getResourceAsStream("/book1.txt");
        String book = IOUtils.toString(defaultConfigStream);
        //ConfigurationService.updateProp("orion-llm4j-llama-inference.maximum.tokens.to.produce", "1048576");
        LLMResponse response = llm.runLLM("Answer in no more than 50 words. Summarise the given book", book, 512);
    }


    @Test
    void test_invalidMaximumTokens()
    {
        assertThrows(InvalidMaximumTokensOptionException.class, () -> {
            LLMResponse response = llm.runLLM("Answer in no more than 12 words. Start your answer with the words \"The sky appears blue, because\"", "Why is the sky blue?", Integer.MAX_VALUE);
        });
    }


    @Test
    void test_invalidUserPrompt()
    {
        assertThrows(InvalidUserPromptException.class, () -> {
            LLMResponse response = llm.runLLM("", "", 512);
        });
    }
}
