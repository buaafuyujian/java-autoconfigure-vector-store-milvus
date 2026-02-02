package com.fyj.rag.vectorstore;

import lombok.Builder;
import lombok.Getter;
import lombok.Singular;

import java.util.List;
import java.util.Map;

/**
 * 向量搜索请求（支持泛型）
 * <p>
 * 使用 Builder 模式构建搜索请求，支持链式调用，泛型指定返回文档类型
 * <p>
 * 示例用法：
 * <pre>{@code
 * // 使用向量搜索，指定返回类型
 * SearchRequest<FaqDocument> request = SearchRequest.<FaqDocument>builder()
 *     .vector(embeddingVector)
 *     .topK(10)
 *     .filter("category == 'tech'")
 *     .documentClass(FaqDocument.class)
 *     .build();
 *
 * // 使用文本搜索（需要配置 EmbeddingModel）
 * SearchRequest<DocumentSegment> request = SearchRequest.<DocumentSegment>builder()
 *     .query("什么是人工智能")
 *     .topK(5)
 *     .documentClass(DocumentSegment.class)
 *     .build();
 *
 * // 默认返回 Document 类型
 * SearchRequest<Document> request = SearchRequest.<Document>builder()
 *     .query("问题")
 *     .topK(10)
 *     .build();
 *
 * List<SearchResult<FaqDocument>> results = vectorStore.similaritySearch(request);
 * }</pre>
 *
 * @param <T> 文档类型，默认为 Document
 */
@Getter
@Builder
public class SearchRequest<T extends Document> {

    /**
     * 查询文本（与 vector 二选一，优先使用 vector）
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
     * 分区名称列表
     */
    @Singular("inPartition")
    private List<String> partitionNames;

    /**
     * 输出字段列表
     */
    @Singular("outputField")
    private List<String> outputFields;

    /**
     * 相似度阈值（0.0 ~ 1.0）
     */
    @Builder.Default
    private float similarityThreshold = 0.0f;

    /**
     * 搜索参数（如 nprobe, ef 等）
     */
    @Singular("searchParam")
    private Map<String, Object> searchParams;

    /**
     * 偏移量（用于分页）
     */
    @Builder.Default
    private int offset = 0;

    /**
     * 返回的文档类型
     */
    @Builder.Default
    private Class<? extends Document> documentClass = Document.class;

    // ==================== 静态工厂方法 ====================

    /**
     * 创建简单向量搜索请求
     */
    public static SearchRequest<Document> of(List<Float> vector, int topK) {
        return SearchRequest.<Document>builder()
                .vector(vector)
                .topK(topK)
                .build();
    }

    /**
     * 创建带过滤条件的向量搜索请求
     */
    public static SearchRequest<Document> of(List<Float> vector, int topK, String filter) {
        return SearchRequest.<Document>builder()
                .vector(vector)
                .topK(topK)
                .filter(filter)
                .build();
    }

    /**
     * 创建简单文本搜索请求
     */
    public static SearchRequest<Document> of(String query, int topK) {
        return SearchRequest.<Document>builder()
                .query(query)
                .topK(topK)
                .build();
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
     * 获取文档类型（类型安全的方式）
     */
    @SuppressWarnings("unchecked")
    public Class<T> getDocumentClass() {
        return (Class<T>) documentClass;
    }
}

