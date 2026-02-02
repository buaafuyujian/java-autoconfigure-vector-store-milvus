package com.fyj.rag.vectorstore;

import lombok.Builder;
import lombok.Data;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 向量搜索请求（Spring AI 风格）
 * <p>
 * 使用 Builder 模式构建搜索请求，支持链式调用
 * <p>
 * 示例用法：
 * <pre>{@code
 * // 使用向量搜索
 * SearchRequest request = SearchRequest.builder()
 *     .vector(embeddingVector)
 *     .topK(10)
 *     .filter("category == 'tech'")
 *     .partitionNames(List.of("partition1", "partition2"))
 *     .similarityThreshold(0.7f)
 *     .build();
 *
 * // 使用文本搜索（需要配置 EmbeddingModel）
 * SearchRequest request = SearchRequest.query("什么是人工智能")
 *     .topK(5)
 *     .filter("type == 'article'")
 *     .build();
 *
 * List<SearchResult> results = vectorStore.similaritySearch(request);
 * }</pre>
 */
@Data
@Builder
public class SearchRequest {

    /**
     * 查询文本（与 vector 二选一，优先使用 vector）
     * <p>
     * 使用文本搜索时，需要在 VectorStore 中配置 EmbeddingModel
     */
    private String query;

    /**
     * 查询向量（与 query 二选一）
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
     * 分区名称列表（可选，不指定则搜索所有分区）
     */
    private List<String> partitionNames;

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

    // ==================== 静态工厂方法 ====================

    /**
     * 创建基于向量的搜索请求
     *
     * @param vector 查询向量
     * @return SearchRequestBuilder
     */
    public static SearchRequestBuilder vector(List<Float> vector) {
        return SearchRequest.builder().vector(vector);
    }

    /**
     * 创建基于文本的搜索请求（需要 EmbeddingModel）
     *
     * @param query 查询文本
     * @return SearchRequestBuilder
     */
    public static SearchRequestBuilder query(String query) {
        return SearchRequest.builder().query(query);
    }

    /**
     * 创建简单向量搜索请求
     *
     * @param vector 查询向量
     * @param topK 返回数量
     * @return SearchRequest
     */
    public static SearchRequest of(List<Float> vector, int topK) {
        return SearchRequest.builder()
                .vector(vector)
                .topK(topK)
                .build();
    }

    /**
     * 创建带过滤条件的向量搜索请求
     *
     * @param vector 查询向量
     * @param topK 返回数量
     * @param filter 过滤表达式
     * @return SearchRequest
     */
    public static SearchRequest of(List<Float> vector, int topK, String filter) {
        return SearchRequest.builder()
                .vector(vector)
                .topK(topK)
                .filter(filter)
                .build();
    }

    /**
     * 创建简单文本搜索请求
     *
     * @param query 查询文本
     * @param topK 返回数量
     * @return SearchRequest
     */
    public static SearchRequest of(String query, int topK) {
        return SearchRequest.builder()
                .query(query)
                .topK(topK)
                .build();
    }

    /**
     * 创建带过滤条件的文本搜索请求
     *
     * @param query 查询文本
     * @param topK 返回数量
     * @param filter 过滤表达式
     * @return SearchRequest
     */
    public static SearchRequest of(String query, int topK, String filter) {
        return SearchRequest.builder()
                .query(query)
                .topK(topK)
                .filter(filter)
                .build();
    }

    // ==================== 链式调用方法 ====================

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

    /**
     * 添加单个分区
     *
     * @param partitionName 分区名称
     * @return this
     */
    public SearchRequest inPartition(String partitionName) {
        if (this.partitionNames == null) {
            this.partitionNames = new ArrayList<>();
        }
        this.partitionNames.add(partitionName);
        return this;
    }

    /**
     * 设置多个分区
     *
     * @param partitionNames 分区名称列表
     * @return this
     */
    public SearchRequest inPartitions(List<String> partitionNames) {
        this.partitionNames = partitionNames;
        return this;
    }

    // ==================== 辅助方法 ====================

    /**
     * 判断是否为文本查询
     */
    public boolean isTextQuery() {
        return query != null && !query.isEmpty() && (vector == null || vector.isEmpty());
    }

    /**
     * 判断是否指定了分区
     */
    public boolean hasPartitions() {
        return partitionNames != null && !partitionNames.isEmpty();
    }

    /**
     * 获取单个分区名（如果只有一个分区）
     */
    public String getPartitionName() {
        if (partitionNames != null && partitionNames.size() == 1) {
            return partitionNames.get(0);
        }
        return null;
    }
}

