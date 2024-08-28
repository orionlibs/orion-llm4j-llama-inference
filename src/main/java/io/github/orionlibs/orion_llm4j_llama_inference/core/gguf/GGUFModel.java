package io.github.orionlibs.orion_llm4j_llama_inference.core.gguf;

import io.github.orionlibs.orion_llm4j_llama_inference.core.tensor.FloatTensor;
import io.github.orionlibs.orion_llm4j_llama_inference.core.tensor.GGUFTensorEntry;
import io.github.orionlibs.orion_llm4j_llama_inference.core.tensor.GGUFTensorInfo;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

public final class GGUFModel
{
    private static final int DEFAULT_ALIGNMENT = 32;
    private int tensorCount;
    private int alignment;
    private Map<String, Object> metadata;
    private Map<String, GGUFTensorInfo> tensorInfos;
    private long tensorDataOffset;
    private MemorySegment tensorData;
    private Map<String, GGUFTensorEntry> tensorEntries;
    private GGUFModelPrimitiveReader primitiveReader;
    private GGUFModelStringReader stringReader;
    private GGUFModelHeaderReader headerReader;


    public GGUFModel()
    {
        this.primitiveReader = new GGUFModelPrimitiveReader();
        this.stringReader = new GGUFModelStringReader();
        this.headerReader = new GGUFModelHeaderReader();
    }


    public static GGUFModel loadModel(Path modelPath) throws IOException
    {
        try(FileChannel fileChannel = FileChannel.open(modelPath))
        {
            GGUFModel gguf = new GGUFModel();
            gguf.loadModelImpl(fileChannel);
            return gguf;
        }
    }


    private void loadModelImpl(FileChannel fileChannel) throws IOException
    {
        headerReader.readHeader(fileChannel);
        this.tensorCount = headerReader.getTensorCount();
        this.metadata = headerReader.getMetadata();
        this.tensorInfos = HashMap.newHashMap(tensorCount);
        populateTensorInfos(fileChannel);
        long _padding = getAlignment() - (fileChannel.position() % getAlignment());
        fileChannel.position(fileChannel.position() + _padding);
        this.tensorDataOffset = fileChannel.position();
        Arena arena = Arena.ofAuto();
        this.tensorData = fileChannel.map(FileChannel.MapMode.READ_ONLY, tensorDataOffset, fileChannel.size() - tensorDataOffset, arena);
        this.tensorEntries = HashMap.newHashMap(tensorInfos.size());
        populateTensorEntries();
    }


    private void populateTensorEntries()
    {
        for(Map.Entry<String, GGUFTensorInfo> entry : tensorInfos.entrySet())
        {
            GGUFTensorInfo ti = entry.getValue();
            int numberOfElements = FloatTensor.numberOfElements(ti.dimensions());
            int sizeInBytes = Math.toIntExact(ti.ggmlType().byteSizeFor(numberOfElements));
            MemorySegment memorySegment = tensorData.asSlice(ti.offset(), sizeInBytes);
            tensorEntries.put(ti.name(), new GGUFTensorEntry(tensorData, ti.name(), ti.ggmlType(), ti.dimensions(), memorySegment));
        }
    }


    private void populateTensorInfos(FileChannel fileChannel) throws IOException
    {
        for(int i = 0; i < tensorCount; ++i)
        {
            GGUFTensorInfo ti = readTensorInfo(fileChannel);
            assert !tensorInfos.containsKey(ti.name());
            tensorInfos.put(ti.name(), ti);
        }
    }


    private GGUFType readGGMLType(FileChannel fileChannel) throws IOException
    {
        int ggmlTypeId = primitiveReader.readInt(fileChannel);
        return GGUFType.fromId(ggmlTypeId);
    }


    private GGUFTensorInfo readTensorInfo(FileChannel fileChannel) throws IOException
    {
        String name = stringReader.readString(fileChannel);
        assert name.length() <= 64;
        int n_dimensions = primitiveReader.readInt(fileChannel);
        assert n_dimensions <= 4;
        int[] dimensions = new int[n_dimensions];
        for(int i = 0; i < n_dimensions; ++i)
        {
            dimensions[i] = Math.toIntExact(primitiveReader.readLong(fileChannel));
        }
        GGUFType ggmlType = readGGMLType(fileChannel);
        long offset = primitiveReader.readLong(fileChannel);
        assert offset % getAlignment() == 0;
        return new GGUFTensorInfo(name, dimensions, ggmlType, offset);
    }


    public int getAlignment()
    {
        if(alignment != 0)
        {
            return alignment;
        }
        alignment = (int)metadata.getOrDefault("general.alignment", DEFAULT_ALIGNMENT);
        assert Integer.bitCount(alignment) == 1 : "alignment must be a power of two";
        return alignment;
    }


    public Map<String, Object> getMetadata()
    {
        return metadata;
    }


    public Map<String, GGUFTensorEntry> getTensorEntries()
    {
        return tensorEntries;
    }
}
