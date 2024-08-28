package io.github.orionlibs.orion_llm4j_llama_inference.core.gguf;

import io.github.orionlibs.orion_llm4j_llama_inference.core.utils.MetadataValueType;
import java.io.IOException;
import java.nio.channels.FileChannel;

final class GGUFModelArrayReader
{
    private GGUFModelPrimitiveReader primitiveReader;
    private GGUFModelStringReader stringReader;
    private GGUFModelMetadataValueTypeReader metadataValueTypeReader;


    GGUFModelArrayReader()
    {
        this.primitiveReader = new GGUFModelPrimitiveReader();
        this.stringReader = new GGUFModelStringReader();
        this.metadataValueTypeReader = new GGUFModelMetadataValueTypeReader();
    }


    Object readArray(FileChannel fileChannel) throws IOException
    {
        MetadataValueType value_type = metadataValueTypeReader.readMetadataValueType(fileChannel);
        int len = Math.toIntExact(primitiveReader.readLong(fileChannel));
        switch(value_type)
        {
            case UINT8, INT8 ->
            {
                return readBytes(fileChannel, len);
            }
            case UINT16, INT16 ->
            {
                return readShorts(fileChannel, len);
            }
            case UINT32, INT32 ->
            {
                return readIntegers(fileChannel, len);
            }
            case FLOAT32 ->
            {
                return readFloats(fileChannel, len);
            }
            case BOOL ->
            {
                return readBooleans(fileChannel, len);
            }
            case STRING ->
            {
                return readStrings(fileChannel, len);
            }
            case ARRAY ->
            {
                return readArrays(fileChannel, len);
            }
            default -> throw new UnsupportedOperationException("read array of " + value_type);
        }
    }


    private Object[] readArrays(FileChannel fileChannel, int len) throws IOException
    {
        Object[] arrays = new Object[len];
        for(int i = 0; i < len; ++i)
        {
            arrays[i] = readArray(fileChannel);
        }
        return arrays;
    }


    private String[] readStrings(FileChannel fileChannel, int len) throws IOException
    {
        String[] strings = new String[len];
        for(int i = 0; i < len; ++i)
        {
            strings[i] = stringReader.readString(fileChannel);
        }
        return strings;
    }


    private boolean[] readBooleans(FileChannel fileChannel, int len) throws IOException
    {
        boolean[] booleans = new boolean[len];
        for(int i = 0; i < len; ++i)
        {
            booleans[i] = primitiveReader.readBoolean(fileChannel);
        }
        return booleans;
    }


    private float[] readFloats(FileChannel fileChannel, int len) throws IOException
    {
        float[] floats = new float[len];
        for(int i = 0; i < len; ++i)
        {
            floats[i] = primitiveReader.readFloat(fileChannel);
        }
        return floats;
    }


    private int[] readIntegers(FileChannel fileChannel, int len) throws IOException
    {
        int[] ints = new int[len];
        for(int i = 0; i < len; ++i)
        {
            ints[i] = primitiveReader.readInt(fileChannel);
        }
        return ints;
    }


    private short[] readShorts(FileChannel fileChannel, int len) throws IOException
    {
        short[] shorts = new short[len];
        for(int i = 0; i < len; ++i)
        {
            shorts[i] = primitiveReader.readShort(fileChannel);
        }
        return shorts;
    }


    private byte[] readBytes(FileChannel fileChannel, int len) throws IOException
    {
        byte[] bytes = new byte[len];
        for(int i = 0; i < len; ++i)
        {
            bytes[i] = primitiveReader.readByte(fileChannel);
        }
        return bytes;
    }
}
