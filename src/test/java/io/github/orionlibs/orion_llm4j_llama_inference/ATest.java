package io.github.orionlibs.orion_llm4j_llama_inference;

import java.io.IOException;
import java.util.TimeZone;
import org.apache.commons.io.IOUtils;

public class ATest
{
    static
    {
        TimeZone.setDefault(TimeZone.getTimeZone("UTC"));
        System.setProperty("active.execution.profile", OrionDomain.testing);
    }

    protected String loadJSONTestResource(String fileLocation)
    {
        try
        {
            return IOUtils.toString(this.getClass().getResourceAsStream(fileLocation));
        }
        catch(IOException e)
        {
            return "";
        }
    }
}
