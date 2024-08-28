package io.github.orionlibs.orion_llm4j_llama_inference.core.gguf;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;

public final class GGUFModelPrimitiveReader
{
    private final ByteBuffer BB_1 = ByteBuffer.allocate(Byte.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    private final ByteBuffer BB_2 = ByteBuffer.allocate(Short.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    private final ByteBuffer BB_4 = ByteBuffer.allocate(Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    private final ByteBuffer BB_8 = ByteBuffer.allocate(Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);


    public byte readByte(FileChannel fileChannel) throws IOException
    {
        int bytesRead = fileChannel.read(BB_1);
        assert bytesRead == 1;
        return BB_1.clear().get(0);
    }


    public boolean readBoolean(FileChannel fileChannel) throws IOException
    {
        return readByte(fileChannel) != 0;
    }


    public short readShort(FileChannel fileChannel) throws IOException
    {
        int bytesRead = fileChannel.read(BB_2);
        assert bytesRead == 2;
        return BB_2.clear().getShort(0);
    }


    public int readInt(FileChannel fileChannel) throws IOException
    {
        int bytesRead = fileChannel.read(BB_4);
        assert bytesRead == 4;
        return BB_4.clear().getInt(0);
    }


    public long readLong(FileChannel fileChannel) throws IOException
    {
        int bytesRead = fileChannel.read(BB_8);
        assert bytesRead == 8;
        return BB_8.clear().getLong(0);
    }


    public float readFloat(FileChannel fileChannel) throws IOException
    {
        return Float.intBitsToFloat(readInt(fileChannel));
    }


    public double readDouble(FileChannel fileChannel) throws IOException
    {
        return Double.longBitsToDouble(readLong(fileChannel));
    }
}
