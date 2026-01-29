package com.fyj.rag.vectorstore;

import lombok.Builder;
import lombok.Data;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 向量搜索请求
 */
@Data
@Builder
public class SearchRequest {

    /**
     * 查询向量
     */
    private List<Float> vector;

    /**
     * 向量字段名称
     */
    @Builder.Default
    private String vectorFieldName = "embedding";

    /**
     * 返回结果数量
     */
    @Builder.Default
    private int topK = 10;

    /**
     * 过滤表达式
     */
    private String filter;

    /**
     * 输出字段列表
     */
    private List<String> outputFields;

    /**
     * 相似度阈值（0.0 ~ 1.0）
     */
    @Builder.Default
    private float similarityThreshold = 0.0f;

    /**
     * 搜索参数（如 nprobe, ef 等）
     */
    @Builder.Default
    private Map<String, Object> searchParams = new HashMap<>();

    /**
     * 偏移量（用于分页）
     */
    @Builder.Default
    private int offset = 0;

    /**
     * 创建简单搜索请求
     */
    public static SearchRequest of(List<Float> vector, int topK) {
        return SearchRequest.builder()
                .vector(vector)
                .topK(topK)
                .build();
    }

    /**
     * 创建带过滤条件的搜索请求
     */
    public static SearchRequest of(List<Float> vector, int topK, String filter) {
        return SearchRequest.builder()
                .vector(vector)
                .topK(topK)
                .filter(filter)
                .build();
    }

    /**
     * 添加搜索参数
     */
    public SearchRequest addSearchParam(String key, Object value) {
        if (this.searchParams == null) {
            this.searchParams = new HashMap<>();
        }
        this.searchParams.put(key, value);
        return this;
    }

    /**
     * 设置 IVF 索引的 nprobe 参数
     */
    public SearchRequest nprobe(int nprobe) {
        return addSearchParam("nprobe", nprobe);
    }

    /**
     * 设置 HNSW 索引的 ef 参数
     */
    public SearchRequest ef(int ef) {
        return addSearchParam("ef", ef);
    }
}

