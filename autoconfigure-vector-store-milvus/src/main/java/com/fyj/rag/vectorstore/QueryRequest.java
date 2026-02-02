package com.fyj.rag.vectorstore;

import lombok.Builder;
import lombok.Data;

import java.util.List;

/**
 * 查询请求（Spring AI 风格）
 * <p>
 * 使用 Builder 模式构建查询请求，支持链式调用
 * <p>
 * 示例用法：
 * <pre>{@code
 * QueryRequest request = QueryRequest.builder()
 *     .filterExpression("age > 18")
 *     .partitionName("user_partition")
 *     .offset(0)
 *     .limit(100)
 *     .build();
 *
 * List<Document> results = vectorStore.query(request);
 * }</pre>
 */
@Data
@Builder
public class QueryRequest {

    /**
     * 过滤表达式
     */
    private String filterExpression;

    /**
     * 分区名称（可选，不指定则查询所有分区）
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
     * 输出字段列表（可选，不指定则返回所有字段）
     */
    private List<String> outputFields;

    /**
     * 创建简单查询请求
     *
     * @param filterExpression 过滤表达式
     * @return 查询请求
     */
    public static QueryRequest filter(String filterExpression) {
        return QueryRequest.builder()
                .filterExpression(filterExpression)
                .build();
    }

    /**
     * 创建带分页的查询请求
     *
     * @param filterExpression 过滤表达式
     * @param offset 偏移量
     * @param limit 限制数量
     * @return 查询请求
     */
    public static QueryRequest of(String filterExpression, int offset, int limit) {
        return QueryRequest.builder()
                .filterExpression(filterExpression)
                .offset(offset)
                .limit(limit)
                .build();
    }

    /**
     * 创建指定分区的查询请求
     *
     * @param filterExpression 过滤表达式
     * @param partitionName 分区名称
     * @return 查询请求
     */
    public static QueryRequest inPartition(String filterExpression, String partitionName) {
        return QueryRequest.builder()
                .filterExpression(filterExpression)
                .partitionName(partitionName)
                .build();
    }
}

