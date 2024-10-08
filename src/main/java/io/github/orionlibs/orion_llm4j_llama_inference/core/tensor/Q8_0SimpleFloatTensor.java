package io.github.orionlibs.orion_llm4j_llama_inference.core.tensor;

import io.github.orionlibs.orion_llm4j_inference.core.gguf.GGUFType;
import io.github.orionlibs.orion_llm4j_inference.core.tensor.FloatTensor;
import io.github.orionlibs.orion_llm4j_inference.core.utils.Float16;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public final class Q8_0SimpleFloatTensor extends SimpleFloatTensor
{
    public static final ValueLayout.OfShort JAVA_SHORT_LE = ValueLayout.JAVA_SHORT.withOrder(ByteOrder.LITTLE_ENDIAN);
    final int size;
    final MemorySegment memorySegment;


    public Q8_0SimpleFloatTensor(int size, MemorySegment memorySegment)
    {
        this.size = size;
        this.memorySegment = memorySegment;
    }


    private static float vectorDot(Q8_0SimpleFloatTensor thiz, int thisOffset, FloatTensor that, int thatOffset, int size)
    {
        float result = 0f;
        int j = 0;
        // Align thisOffset + startIndex to type().getBlockSize().
        assert Integer.bitCount(GGUFType.Q8_0.getBlockSize()) == 1 : "power of 2";
        int alignmentBound = Math.min(size, -thisOffset & (GGUFType.Q8_0.getBlockSize() - 1));
        if(alignmentBound > 0)
        {
            result += that.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }
        assert (thisOffset + j) % GGUFType.Q8_0.getBlockSize() == 0;
        FloatVector val = FloatVector.zero(F_SPECIES);
        int blockOffset = (thisOffset + j) / GGUFType.Q8_0.getBlockSize() * GGUFType.Q8_0.getTypeSize();
        int upperBound = size / GGUFType.Q8_0.getBlockSize() * GGUFType.Q8_0.getBlockSize();
        for(; j < upperBound; j += GGUFType.Q8_0.getBlockSize(), blockOffset += GGUFType.Q8_0.getTypeSize())
        {
            float wScaleValue = Float.float16ToFloat(thiz.memorySegment.get(JAVA_SHORT_LE, blockOffset));
            var wScale = FloatVector.broadcast(F_SPECIES, wScaleValue);
            if(F_SPECIES.vectorBitSize() == 256)
            {
                var wBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_256, thiz.memorySegment, blockOffset + Float16.BYTES, ByteOrder.LITTLE_ENDIAN);
                var sum0 = that.getFloatVector(F_SPECIES, thatOffset + j + 0 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 0));
                var sum1 = that.getFloatVector(F_SPECIES, thatOffset + j + 1 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 1));
                var sum2 = that.getFloatVector(F_SPECIES, thatOffset + j + 2 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 2));
                var sum3 = that.getFloatVector(F_SPECIES, thatOffset + j + 3 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 3));
                val = sum0.add(sum1).add(sum2).add(sum3).fma(wScale, val);
            }
            else if(F_SPECIES.vectorBitSize() == 128)
            {
                VectorSpecies<Byte> B_128 = ByteVector.SPECIES_128;
                // This loop cannot be unrolled, why?
                for(int i = 0; i < 2; i++)
                {
                    var wBytes = ByteVector.fromMemorySegment(B_128, thiz.memorySegment, blockOffset + Float16.BYTES + i * B_128.vectorByteSize(), ByteOrder.LITTLE_ENDIAN);
                    var sum0 = that.getFloatVector(F_SPECIES, thatOffset + j + i * 16 + 0 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 0));
                    var sum1 = that.getFloatVector(F_SPECIES, thatOffset + j + i * 16 + 1 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 1));
                    var sum2 = that.getFloatVector(F_SPECIES, thatOffset + j + i * 16 + 2 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 2));
                    var sum3 = that.getFloatVector(F_SPECIES, thatOffset + j + i * 16 + 3 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 3));
                    val = sum0.add(sum1).add(sum2).add(sum3).fma(wScale, val);
                }
            }
            else
            {
                throw new UnsupportedOperationException(F_SPECIES.toString());
            }
        }
        result += val.reduceLanes(VectorOperators.ADD);
        // Remaining entries.
        if(j < size)
        {
            result += that.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }
        return result;
    }


    @Override
    public int size()
    {
        return size;
    }


    @Override
    public void setFloat(int index, float value)
    {
        throw new UnsupportedOperationException("setFloat");
    }


    @Override
    public FloatVector getFloatVector(VectorSpecies<Float> species, int index)
    {
        throw new UnsupportedOperationException("getFloatVector");
    }


    @Override
    public GGUFType type()
    {
        return GGUFType.Q8_0;
    }


    @Override
    public float getFloat(int index)
    {
        assert 0 <= index && index < size;
        int blockIndex = index / GGUFType.Q8_0.getBlockSize();
        int withinBlockIndex = index % GGUFType.Q8_0.getBlockSize();
        int blockOffset = blockIndex * GGUFType.Q8_0.getTypeSize();
        byte quant = memorySegment.get(ValueLayout.JAVA_BYTE, blockOffset + Float16.BYTES + withinBlockIndex);
        float scale = Float.float16ToFloat(memorySegment.get(JAVA_SHORT_LE, blockOffset));
        return quant * scale;
    }


    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size)
    {
        if(SimpleFloatTensor.USE_VECTOR_API)
        {
            return vectorDot(this, thisOffset, that, thatOffset, size);
        }
        else
        {
            return that.scalarDot(this, thisOffset, that, thatOffset, size);
        }
    }
}
