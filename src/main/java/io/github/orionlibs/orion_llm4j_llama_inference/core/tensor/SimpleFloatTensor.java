package io.github.orionlibs.orion_llm4j_llama_inference.core.tensor;

import io.github.orionlibs.orion_llm4j_inference.core.gguf.GGUFType;
import io.github.orionlibs.orion_llm4j_inference.core.tensor.FloatTensor;
import io.github.orionlibs.orion_llm4j_inference.core.utils.AggregateFunction;
import io.github.orionlibs.orion_llm4j_inference.core.utils.MapFunction;
import io.github.orionlibs.orion_llm4j_inference.core.utils.MapWithIndexFunction;
import io.github.orionlibs.orion_llm4j_inference.core.utils.Parallel;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Over-simplified, shapeless, float tensor.
 * <p>
 * Not a strict tensor, but rather just a sequence of floats, not required to be backed by memory
 * e.g. can represent a sequence of quantized floats.
 */
public abstract class SimpleFloatTensor implements FloatTensor
{
    static final ValueLayout.OfFloat JAVA_FLOAT_LE = ValueLayout.JAVA_FLOAT.withOrder(ByteOrder.LITTLE_ENDIAN);
    static final ValueLayout.OfShort JAVA_SHORT_LE = ValueLayout.JAVA_SHORT.withOrder(ByteOrder.LITTLE_ENDIAN);
    static final boolean USE_VECTOR_API = Boolean.parseBoolean(System.getProperty("llama.VectorAPI", "true"));
    // Preferred vector size for the fast multiplication routines.
    // (Apple Silicon) NEON only supports up-to 128bit vectors.
    static final VectorSpecies<Float> F_SPECIES = FloatVector.SPECIES_PREFERRED.vectorBitSize() == 128 ? FloatVector.SPECIES_128 : FloatVector.SPECIES_256;


    public abstract int size();


    public abstract float getFloat(int index);


    public abstract void setFloat(int index, float value);


    public abstract FloatVector getFloatVector(VectorSpecies<Float> species, int offset);


    public abstract GGUFType type();


    @Override
    public float scalarDot(FloatTensor thiz, int thisOffset, FloatTensor that, int thatOffset, int size)
    {
        float result = 0f;
        for(int j = 0; j < size; j++)
        {
            result += thiz.getFloat(thisOffset + j) * that.getFloat(thatOffset + j);
        }
        return result;
    }


    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size)
    {
        return scalarDot(this, thisOffset, that, thatOffset, size);
    }


    @Override
    public void matmul(FloatTensor that, FloatTensor out, int dim0, int dim1)
    {
        Parallel.parallelFor(0, dim0, i -> out.setFloat(i, dot(i * dim1, that, 0, dim1)));
    }


    @Override
    public float reduce(int thisOffset, int size, float seed, AggregateFunction reduce)
    {
        float result = seed;
        for(int i = 0; i < size; ++i)
        {
            result = reduce.apply(result, getFloat(thisOffset + i));
        }
        return result;
    }


    @Override
    public float sum(int thisOffset, int size)
    {
        return reduce(thisOffset, size, 0f, Float::sum);
    }


    @Override
    public float max(int thisOffset, int size)
    {
        return reduce(thisOffset, size, Float.NEGATIVE_INFINITY, Float::max);
    }


    @Override
    public void copyTo(int thisOffset, FloatTensor that, int thatOffset, int size)
    {
        that.mapWithIndexInPlace(thatOffset, size, (value, index) -> this.getFloat(index - thatOffset + thisOffset));
    }


    @Override
    public int argmax(int thisOffset, int size)
    {
        assert size > 0;
        int maxIndex = thisOffset;
        float maxValue = this.getFloat(maxIndex);
        int endIndex = thisOffset + size;
        for(int i = thisOffset; i < endIndex; ++i)
        {
            float f = this.getFloat(i);
            if(f > maxValue)
            {
                maxValue = f;
                maxIndex = i;
            }
        }
        return maxIndex;
    }


    @Override
    public int argmax()
    {
        return argmax(0, size());
    }


    @Override
    public SimpleFloatTensor mapInPlace(int thisOffset, int size, MapFunction mapFunction)
    {
        int endIndex = thisOffset + size;
        for(int i = thisOffset; i < endIndex; ++i)
        {
            setFloat(i, mapFunction.apply(getFloat(i)));
        }
        return this;
    }


    @Override
    public SimpleFloatTensor mapInPlace(MapFunction mapFunction)
    {
        return mapInPlace(0, size(), mapFunction);
    }


    @Override
    public SimpleFloatTensor mapWithIndexInPlace(int thisOffset, int size, MapWithIndexFunction mapWithIndexFunction)
    {
        int endOffset = thisOffset + size;
        for(int i = thisOffset; i < endOffset; ++i)
        {
            setFloat(i, mapWithIndexFunction.apply(getFloat(i), i));
        }
        return this;
    }


    @Override
    public SimpleFloatTensor addInPlace(int thisOffset, FloatTensor that, int thatOffset, int size)
    {
        return mapWithIndexInPlace(thisOffset, size, (value, index) -> value + that.getFloat(index - thisOffset + thatOffset));
    }


    @Override
    public SimpleFloatTensor addInPlace(FloatTensor that)
    {
        return addInPlace(0, that, 0, size());
    }


    @Override
    public SimpleFloatTensor multiplyInPlace(int thisOffset, FloatTensor that, int thatOffset, int size)
    {
        return mapWithIndexInPlace(thisOffset, size, (value, index) -> value * that.getFloat(index - thisOffset + thatOffset));
    }


    @Override
    public SimpleFloatTensor multiplyInPlace(FloatTensor that)
    {
        return multiplyInPlace(0, that, 0, size());
    }


    @Override
    public SimpleFloatTensor divideInPlace(int thisOffset, int size, float value)
    {
        return mapInPlace(thisOffset, size, f -> f / value);
    }


    @Override
    public SimpleFloatTensor fillInPlace(int thisOffset, int size, float value)
    {
        return mapInPlace(thisOffset, size, unused -> value);
    }


    @Override
    public SimpleFloatTensor softmaxInPlace(int thisOffset, int size)
    {
        // find max value (for numerical stability)
        float maxVal = max(thisOffset, size);
        // exp and sum
        mapInPlace(thisOffset, size, f -> (float)Math.exp(f - maxVal));
        float sum = sum(thisOffset, size);
        // normalize
        return divideInPlace(thisOffset, size, sum);
    }


    @Override
    public SimpleFloatTensor saxpyInPlace(int thisOffset, FloatTensor that, int thatOffset, int size, float a)
    {
        // this[thatOffset ... thatOffset + size) = a * that[thatOffset ... thatOffset + size) + this[thisOffset ... thisOffset + size)
        for(int i = 0; i < size; ++i)
        {
            setFloat(thisOffset + i, a * that.getFloat(thatOffset + i) + this.getFloat(thisOffset + i));
        }
        return this;
    }
}
