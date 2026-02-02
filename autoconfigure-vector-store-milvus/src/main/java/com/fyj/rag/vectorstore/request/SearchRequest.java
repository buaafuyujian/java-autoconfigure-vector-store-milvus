package com.fyj.rag.vectorstore.request;

import com.fyj.rag.vectorstore.model.Document;
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
 * 支持三种搜索类型：
 * <ul>
 *     <li>{@link SearchType#VECTOR} - 向量相似度搜索（默认）</li>
 *     <li>{@link SearchType#BM25} - BM25 全文检索</li>
 *     <li>{@link SearchType#HYBRID} - 混合搜索（向量 + BM25）</li>
 * </ul>
 * <p>
 * 示例用法：
 * <pre>{@code
 * // 1. 向量搜索，指定返回类型
 * SearchRequest<FaqDocument> request = SearchRequest.<FaqDocument>builder()
 *     .vector(embeddingVector)
 *     .topK(10)
 *     .filter("category == 'tech'")
 *     .documentClass(FaqDocument.class)
 *     .build();
 *
 * // 2. 使用文本进行向量搜索（需要配置 EmbeddingModel）
 * SearchRequest<DocumentSegment> request = SearchRequest.<DocumentSegment>builder()
 *     .query("什么是人工智能")
 *     .topK(5)
 *     .documentClass(DocumentSegment.class)
 *     .build();
 *
 * // 3. BM25 全文检索
 * SearchRequest<Document> bm25Request = SearchRequest.<Document>builder()
 *     .query("人工智能 机器学习")
 *     .searchType(SearchType.BM25)
 *     .textFieldName("content")
 *     .topK(10)
 *     .build();
 *
 * // 4. 混合搜索（向量 + BM25）
 * SearchRequest<Document> hybridRequest = SearchRequest.<Document>builder()
 *     .query("什么是深度学习")
 *     .searchType(SearchType.HYBRID)
 *     .vectorWeight(0.7f)
 *     .bm25Weight(0.3f)
 *     .topK(10)
 *     .build();
 *
 * List<SearchResult<FaqDocument>> results = vectorStore.search(request);
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
     * 搜索类型（向量搜索、BM25搜索、混合搜索）
     * <p>
     * 默认为向量搜索（VECTOR）
     */
    @Builder.Default
    private SearchType searchType = SearchType.VECTOR;

    /**
     * BM25 搜索的文本字段名称
     * <p>
     * 用于 BM25 和混合搜索时指定搜索的文本字段
     */
    @Builder.Default
    private String textFieldName = "content";

    /**
     * 稀疏向量字段名称
     * <p>
     * 用于 BM25 和混合搜索时指定稀疏向量字段
     */
    @Builder.Default
    private String sparseVectorFieldName = "sparse";

    /**
     * 混合搜索时向量搜索的权重（0.0 ~ 1.0）
     * <p>
     * 仅在 searchType 为 HYBRID 时生效
     * 默认值 0.5 表示向量搜索和 BM25 各占 50% 权重
     */
    @Builder.Default
    private float vectorWeight = 0.5f;

    /**
     * 混合搜索时 BM25 搜索的权重（0.0 ~ 1.0）
     * <p>
     * 仅在 searchType 为 HYBRID 时生效
     * 默认值 0.5 表示向量搜索和 BM25 各占 50% 权重
     */
    @Builder.Default
    private float bm25Weight = 0.5f;

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

    /**
     * 创建 BM25 全文检索请求
     *
     * @param query 查询文本
     * @param topK 返回结果数量
     */
    public static SearchRequest<Document> bm25(String query, int topK) {
        return SearchRequest.<Document>builder()
                .query(query)
                .topK(topK)
                .searchType(SearchType.BM25)
                .build();
    }

    /**
     * 创建 BM25 全文检索请求（指定文本字段）
     *
     * @param query 查询文本
     * @param topK 返回结果数量
     * @param textFieldName 文本字段名称
     */
    public static SearchRequest<Document> bm25(String query, int topK, String textFieldName) {
        return SearchRequest.<Document>builder()
                .query(query)
                .topK(topK)
                .searchType(SearchType.BM25)
                .textFieldName(textFieldName)
                .build();
    }

    /**
     * 创建混合搜索请求（默认权重各 50%）
     *
     * @param query 查询文本
     * @param topK 返回结果数量
     */
    public static SearchRequest<Document> hybrid(String query, int topK) {
        return SearchRequest.<Document>builder()
                .query(query)
                .topK(topK)
                .searchType(SearchType.HYBRID)
                .build();
    }

    /**
     * 创建混合搜索请求（自定义权重）
     *
     * @param query 查询文本
     * @param topK 返回结果数量
     * @param vectorWeight 向量搜索权重
     * @param bm25Weight BM25 搜索权重
     */
    public static SearchRequest<Document> hybrid(String query, int topK, float vectorWeight, float bm25Weight) {
        return SearchRequest.<Document>builder()
                .query(query)
                .topK(topK)
                .searchType(SearchType.HYBRID)
                .vectorWeight(vectorWeight)
                .bm25Weight(bm25Weight)
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
     * 判断是否为向量搜索模式
     */
    public boolean isVectorSearch() {
        return searchType == SearchType.VECTOR;
    }

    /**
     * 判断是否为 BM25 搜索模式
     */
    public boolean isBm25Search() {
        return searchType == SearchType.BM25;
    }

    /**
     * 判断是否为混合搜索模式
     */
    public boolean isHybridSearch() {
        return searchType == SearchType.HYBRID;
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

