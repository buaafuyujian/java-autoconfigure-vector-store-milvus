package com.fyj.rag.vectorstore;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 文档实体
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class Document {

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
    @Builder.Default
    private Map<String, Object> metadata = new HashMap<>();

    /**
     * 创建文档
     */
    public static Document of(String id, String content, List<Float> embedding) {
        return Document.builder()
                .id(id)
                .content(content)
                .embedding(embedding)
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
}

