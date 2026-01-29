package com.fyj.rag.vectorstore;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * 向量搜索结果
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class SearchResult {

    /**
     * 匹配的文档
     */
    private Document document;

    /**
     * 相似度分数（归一化到 0-1，越高越相似）
     */
    private float score;

    /**
     * 原始距离值
     */
    private float distance;

    /**
     * 创建搜索结果
     */
    public static SearchResult of(Document document, float score, float distance) {
        return SearchResult.builder()
                .document(document)
                .score(score)
                .distance(distance)
                .build();
    }

    /**
     * 创建搜索结果（仅分数）
     */
    public static SearchResult of(Document document, float score) {
        return SearchResult.builder()
                .document(document)
                .score(score)
                .distance(score)
                .build();
    }
}

