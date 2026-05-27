package com.fyj.rag.vectorstore.request;

import com.fyj.rag.vectorstore.model.Document;
import lombok.Getter;
import lombok.Singular;
import lombok.experimental.SuperBuilder;

import java.util.List;
import java.util.Map;

/**
 * 搜索请求基类，包含单次搜索与批量搜索共用的所有参数
 * <p>
 * 子类：
 * <ul>
 *     <li>{@link SearchRequest} - 单次搜索（单个 query / vector）</li>
 *     <li>{@link BatchSearchRequest} - 批量搜索（多个 queries / vectors，一次 RPC）</li>
 * </ul>
 *
 * @param <T> 文档类型，默认为 Document
 */
@Getter
@SuperBuilder
public abstract class BaseSearchRequest<T extends Document> {

    /**
     * 向量字段名称
     */
    @lombok.Builder.Default
    private String vectorFieldName = "embedding";

    /**
     * 返回结果数量
     */
    @lombok.Builder.Default
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
    @lombok.Builder.Default
    private float similarityThreshold = 0.0f;

    /**
     * 搜索参数（如 nprobe, ef 等）
     */
    @Singular("searchParam")
    private Map<String, Object> searchParams;

    /**
     * 偏移量（用于分页）
     */
    @lombok.Builder.Default
    private int offset = 0;

    /**
     * 搜索类型（向量搜索、BM25搜索、混合搜索），默认 VECTOR
     */
    @lombok.Builder.Default
    private SearchType searchType = SearchType.VECTOR;

    /**
     * BM25 搜索的文本字段名称
     */
    @lombok.Builder.Default
    private String textFieldName = "content";

    /**
     * 稀疏向量字段名称（BM25 / 混合搜索使用）
     */
    @lombok.Builder.Default
    private String sparseVectorFieldName = "sparse";

    /**
     * 混合搜索时向量搜索的权重（0.0 ~ 1.0），默认 0.5
     */
    @lombok.Builder.Default
    private float vectorWeight = 0.5f;

    /**
     * 混合搜索时 BM25 搜索的权重（0.0 ~ 1.0），默认 0.5
     */
    @lombok.Builder.Default
    private float bm25Weight = 0.5f;

    /**
     * 返回的文档类型，默认 Document
     */
    @lombok.Builder.Default
    private Class<? extends Document> documentClass = Document.class;

    // ==================== 辅助方法 ====================

    /**
     * 判断是否指定了分区
     */
    public boolean hasPartitions() {
        return partitionNames != null && !partitionNames.isEmpty();
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
     * 获取文档类型（类型安全）
     */
    @SuppressWarnings("unchecked")
    public Class<T> getDocumentClass() {
        return (Class<T>) documentClass;
    }
}

