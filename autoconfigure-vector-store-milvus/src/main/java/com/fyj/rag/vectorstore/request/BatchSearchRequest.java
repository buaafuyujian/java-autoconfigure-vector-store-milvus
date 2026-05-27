package com.fyj.rag.vectorstore.request;

import com.fyj.rag.vectorstore.MilvusVectorStore;
import com.fyj.rag.vectorstore.model.Document;
import lombok.Getter;
import lombok.experimental.SuperBuilder;

import java.util.Collections;
import java.util.List;

/**
 * 批量搜索请求
 * <p>
 * 通过 Milvus SDK 原生的 {@code SearchReq.data(List<BaseVector>)} 支持，
 * 将多个查询向量/文本打包进同一个请求，一次 RPC 完成，避免多次网络往返。
 * <p>
 * 返回值中每个元素对应一个输入 query 的结果列表，顺序与输入保持一致。
 * <p>
 * 示例用法：
 * <pre>{@code
 * // 多文本批量向量搜索
 * BatchSearchRequest<DocumentSegment> req = BatchSearchRequest.<DocumentSegment>builder()
 *     .queries(List.of("人工智能", "机器学习", "深度学习"))
 *     .topK(3)
 *     .documentClass(DocumentSegment.class)
 *     .build();
 * List<List<SearchResult<DocumentSegment>>> results = vectorStore.batchSearch(req);
 * // results.get(0) -> "人工智能" 的 topK 结果
 * // results.get(1) -> "机器学习" 的 topK 结果
 * // results.get(2) -> "深度学习" 的 topK 结果
 *
 * // 多向量批量搜索
 * BatchSearchRequest<Document> vecReq = BatchSearchRequest.<Document>builder()
 *     .vectors(List.of(vec1, vec2, vec3))
 *     .topK(5)
 *     .build();
 * List<List<SearchResult<Document>>> results = vectorStore.batchSearch(vecReq);
 *
 * // 批量 BM25 搜索
 * BatchSearchRequest<Document> bm25Req = BatchSearchRequest.<Document>builder()
 *     .queries(List.of("框架", "深度学习"))
 *     .searchType(SearchType.BM25)
 *     .topK(3)
 *     .build();
 *
 * // 批量混合搜索
 * BatchSearchRequest<Document> hybridReq = BatchSearchRequest.<Document>builder()
 *     .queries(List.of("人工智能", "机器学习"))
 *     .searchType(SearchType.HYBRID)
 *     .vectorWeight(0.7f)
 *     .bm25Weight(0.3f)
 *     .topK(5)
 *     .build();
 * }</pre>
 *
 * @param <T> 文档类型，默认为 Document
 * @see MilvusVectorStore#batchSearch(BatchSearchRequest)
 */
@Getter
@SuperBuilder
public class BatchSearchRequest<T extends Document> extends BaseSearchRequest<T> {

    /**
     * 批量查询文本列表（{@code SearchRequest#query} 的复数形式）
     * <p>
     * 适用于向量搜索（自动 embed）、BM25 搜索、混合搜索
     */
    private List<String> queries;

    /**
     * 批量查询向量列表（{@code SearchRequest#vector} 的复数形式）
     * <p>
     * 适用于向量搜索（直接使用预计算向量）、混合搜索
     */
    private List<List<Float>> vectors;

    // ==================== 静态工厂方法 ====================

    /**
     * 创建批量向量搜索请求（文本列表，自动 embed）
     */
    public static BatchSearchRequest<Document> ofQueries(List<String> queries, int topK) {
        return BatchSearchRequest.<Document>builder()
                .queries(queries)
                .topK(topK)
                .build();
    }

    /**
     * 创建批量向量搜索请求（预计算向量列表）
     */
    public static BatchSearchRequest<Document> ofVectors(List<List<Float>> vectors, int topK) {
        return BatchSearchRequest.<Document>builder()
                .vectors(vectors)
                .topK(topK)
                .build();
    }

    /**
     * 创建批量 BM25 搜索请求
     */
    public static BatchSearchRequest<Document> bm25(List<String> queries, int topK) {
        return BatchSearchRequest.<Document>builder()
                .queries(queries)
                .topK(topK)
                .searchType(SearchType.BM25)
                .build();
    }

    /**
     * 创建批量混合搜索请求（默认权重各 50%）
     */
    public static BatchSearchRequest<Document> hybrid(List<String> queries, int topK) {
        return BatchSearchRequest.<Document>builder()
                .queries(queries)
                .topK(topK)
                .searchType(SearchType.HYBRID)
                .build();
    }

    /**
     * 创建批量混合搜索请求（自定义权重）
     */
    public static BatchSearchRequest<Document> hybrid(List<String> queries, int topK,
                                                       float vectorWeight, float bm25Weight) {
        return BatchSearchRequest.<Document>builder()
                .queries(queries)
                .topK(topK)
                .searchType(SearchType.HYBRID)
                .vectorWeight(vectorWeight)
                .bm25Weight(bm25Weight)
                .build();
    }

    // ==================== 辅助方法 ====================

    /**
     * 获取有效的查询文本列表
     * <p>
     * 若 {@code queries} 非空则返回之，否则返回空列表
     */
    public List<String> getEffectiveTexts() {
        return (queries != null && !queries.isEmpty()) ? queries : Collections.emptyList();
    }

    /**
     * 获取有效的查询向量列表
     * <p>
     * 若 {@code vectors} 非空则返回之，否则返回空��表
     */
    public List<List<Float>> getEffectiveVectors() {
        return (vectors != null && !vectors.isEmpty()) ? vectors : Collections.emptyList();
    }
}

