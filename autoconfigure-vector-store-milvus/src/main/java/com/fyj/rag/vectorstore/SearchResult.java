package com.fyj.rag.vectorstore;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * 向量搜索结果
 *
 * @param <T> 文档类型，必须是 Document 或其子类
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class SearchResult<T extends Document> {

    /**
     * 匹配的文档
     */
    private T document;

    /**
     * 相似度分数（归一化到 0-1，越高越相似）
     */
    private float score;

    /**
     * 原始距离值
     */
    private float distance;

    /**
     * 创建搜索结果（泛型版本）
     */
    public static <T extends Document> SearchResult<T> of(T document, float score, float distance) {
        return SearchResult.<T>builder()
                .document(document)
                .score(score)
                .distance(distance)
                .build();
    }

    /**
     * 创建搜索结果（仅分数）
     */
    public static <T extends Document> SearchResult<T> of(T document, float score) {
        return SearchResult.<T>builder()
                .document(document)
                .score(score)
                .distance(score)
                .build();
    }
}

