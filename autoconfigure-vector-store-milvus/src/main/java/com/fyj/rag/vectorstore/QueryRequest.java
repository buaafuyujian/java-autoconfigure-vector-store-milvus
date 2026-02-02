package com.fyj.rag.vectorstore;

import lombok.Builder;
import lombok.Getter;
import lombok.Singular;

import java.util.List;

/**
 * 查询请求（支持泛型）
 * <p>
 * 使用 Builder 模式构建查询请求，泛型指定返回文档类型
 * <p>
 * 示例用法：
 * <pre>{@code
 * // 指定返回类型
 * QueryRequest<FaqDocument> request = QueryRequest.<FaqDocument>builder()
 *     .filter("type == 'faq'")
 *     .inPartition("knowledge_base")
 *     .documentClass(FaqDocument.class)
 *     .build();
 * List<FaqDocument> docs = vectorStore.query(request);
 *
 * // 默认返回 Document 类型
 * QueryRequest<Document> request = QueryRequest.<Document>builder()
 *     .filter("category == 'tech'")
 *     .build();
 * List<Document> docs = vectorStore.query(request);
 * }</pre>
 *
 * @param <T> 文档类型，默认为 Document
 */
@Getter
@Builder
public class QueryRequest<T extends Document> {

    /**
     * 过滤表达式
     */
    private String filter;

    /**
     * 分区名称
     */
    private String partitionName;

    /**
     * 偏移量（用于分页）
     */
    @Builder.Default
    private int offset = 0;

    /**
     * 限制数量（用于分页）
     */
    @Builder.Default
    private int limit = 100;

    /**
     * 输出字段列表
     */
    @Singular("outputField")
    private List<String> outputFields;

    /**
     * 返回的文档类型
     */
    @Builder.Default
    private Class<? extends Document> documentClass = Document.class;

    // ==================== 静态工厂方法 ====================

    /**
     * 创建简单查询请求
     */
    public static QueryRequest<Document> of(String filter) {
        return QueryRequest.<Document>builder()
                .filter(filter)
                .build();
    }

    /**
     * 创建带分页的查询请求
     */
    public static QueryRequest<Document> of(String filter, int offset, int limit) {
        return QueryRequest.<Document>builder()
                .filter(filter)
                .offset(offset)
                .limit(limit)
                .build();
    }

    /**
     * 创建指定分区的查询请求
     */
    public static QueryRequest<Document> of(String filter, String partitionName) {
        return QueryRequest.<Document>builder()
                .filter(filter)
                .partitionName(partitionName)
                .build();
    }

    // ==================== 辅助方法 ====================

    /**
     * 获取过滤表达式（兼容旧 API）
     */
    public String getFilterExpression() {
        return filter;
    }

    /**
     * 获取文档类型（类型安全的方式）
     */
    @SuppressWarnings("unchecked")
    public Class<T> getDocumentClass() {
        return (Class<T>) documentClass;
    }
}

