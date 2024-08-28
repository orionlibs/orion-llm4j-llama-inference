package io.github.orionlibs.orion_llm4j_llama_inference;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.orionlibs.orion_llm4j_llama_inference.core.Response;
import java.io.IOException;
import org.junit.jupiter.api.BeforeEach;
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
        Response response = llm.runLLM("Why is the sky blue? Answer in no more than 12 words. Start your answer with the words \"The sky appears blue due to\"");
        String capturedOutput = response.getContent();
        assertTrue(capturedOutput.startsWith("The sky appears blue due to"));
    }
}
