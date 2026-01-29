package com.fyj.rag.vectorstore;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 文档实体
 */
@Data
@SuperBuilder
@NoArgsConstructor
@AllArgsConstructor
public class Document {

    private static final Gson GSON = new Gson();

    /**
     * 文档唯一标识
     */
    private String id;

    /**
     * 文档内容
     */
    private String content;

    /**
     * 向量 embedding
     */
    private List<Float> embedding;

    /**
     * 元数据
     */
    private Map<String, Object> metadata;

    /**
     * 字段名称常量
     */
    public static final String FIELD_ID = "id";
    public static final String FIELD_CONTENT = "content";
    public static final String FIELD_EMBEDDING = "embedding";
    public static final String FIELD_METADATA = "metadata";

    /**
     * 创建文档
     */
    public static Document of(String id, String content, List<Float> embedding) {
        return Document.builder()
                .id(id)
                .content(content)
                .embedding(embedding)
                .metadata(new HashMap<>())
                .build();
    }

    /**
     * 创建文档（带元数据）
     */
    public static Document of(String id, String content, List<Float> embedding, Map<String, Object> metadata) {
        return Document.builder()
                .id(id)
                .content(content)
                .embedding(embedding)
                .metadata(metadata)
                .build();
    }

    /**
     * 添加元数据
     */
    public Document addMetadata(String key, Object value) {
        if (this.metadata == null) {
            this.metadata = new HashMap<>();
        }
        this.metadata.put(key, value);
        return this;
    }

    /**
     * 获取元数据，确保不为 null
     */
    public Map<String, Object> getMetadata() {
        if (this.metadata == null) {
            this.metadata = new HashMap<>();
        }
        return this.metadata;
    }

    /**
     * 转换为 JsonObject，用于插入 Milvus
     * 使用 Gson 序列化，子类的字段会自动包含
     */
    public JsonObject toJsonObject() {
        return GSON.toJsonTree(this).getAsJsonObject();
    }
}

