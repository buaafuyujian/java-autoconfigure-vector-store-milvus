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
     * 子类可以覆盖此方法添加额外字段
     */
    public JsonObject toJsonObject(String idField, String contentField, String embeddingField, String metadataField) {
        JsonObject json = new JsonObject();
        json.addProperty(idField, this.id);

        if (this.content != null) {
            json.addProperty(contentField, this.content);
        }

        if (this.embedding != null) {
            json.add(embeddingField, GSON.toJsonTree(this.embedding));
        }

        if (this.metadata != null && !this.metadata.isEmpty()) {
            json.add(metadataField, GSON.toJsonTree(this.metadata));
        }

        return json;
    }
}

