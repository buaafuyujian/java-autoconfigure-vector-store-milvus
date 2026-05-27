package com.fyj.rag.vectorstore.request;

import com.fyj.rag.vectorstore.model.Document;
import lombok.Getter;
import lombok.experimental.SuperBuilder;

import java.util.List;

/**
 * 单次搜索请求
 * <p>
 * 每次请求包含一个 query 文本或一个 query 向量，对应 Milvus SDK 的单次搜索调用。
 * <p>
 * 如需一次 RPC 同时搜索多个 query，请使用 {@link BatchSearchRequest}。
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
 * SearchRequest<Document> bm25Request = SearchRequest.bm25("人工智能 机器学习", 10);
 *
 * // 4. 混合搜索（向量 + BM25）
 * SearchRequest<Document> hybridRequest = SearchRequest.hybrid("什么是深度学习", 10, 0.7f, 0.3f);
 *
 * List<SearchResult<FaqDocument>> results = vectorStore.search(request);
 * }</pre>
 *
 * @param <T> 文档类型，默认为 Document
 * @see BatchSearchRequest 批量搜索（多 query，一次 RPC）
 */
@Getter
@SuperBuilder
public class SearchRequest<T extends Document> extends BaseSearchRequest<T> {

    /**
     * 查询文本（与 vector 二选一，优先使用 vector）
     */
    private String query;

    /**
     * 查询向量（与 query 二选一）
     */
    private List<Float> vector;

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
     */
    public static SearchRequest<Document> hybrid(String query, int topK,
                                                  float vectorWeight, float bm25Weight) {
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
     * 判断是否为文本查询（有 query 且无 vector）
     */
    public boolean isTextQuery() {
        return query != null && !query.isEmpty() && (vector == null || vector.isEmpty());
    }
}
