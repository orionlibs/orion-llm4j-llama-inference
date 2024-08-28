package io.github.orionlibs.orion_llm4j_llama_inference.core.tensor;

import io.github.orionlibs.orion_llm4j_inference.core.gguf.GGUFType;
import java.util.Arrays;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

public final class ArrayFloatTensor extends FloatTensor
{
    final float[] values;


    ArrayFloatTensor(float[] values)
    {
        this.values = values;
    }


    public static FloatTensor allocate(int... dims)
    {
        int numberOfElements = FloatTensor.numberOfElements(dims);
        return new ArrayFloatTensor(new float[numberOfElements]);
    }


    @Override
    public int size()
    {
        return values.length;
    }


    @Override
    public float getFloat(int index)
    {
        return values[index];
    }


    @Override
    public void setFloat(int index, float value)
    {
        values[index] = value;
    }


    @Override
    public GGUFType type()
    {
        return GGUFType.F32;
    }


    @Override
    public FloatTensor fillInPlace(int thisOffset, int size, float value)
    {
        Arrays.fill(values, thisOffset, thisOffset + size, value);
        return this;
    }


    @Override
    public FloatVector getFloatVector(VectorSpecies<Float> species, int index)
    {
        if(!USE_VECTOR_API)
        {
            throw new UnsupportedOperationException();
        }
        return FloatVector.fromArray(species, values, index);
    }
}
