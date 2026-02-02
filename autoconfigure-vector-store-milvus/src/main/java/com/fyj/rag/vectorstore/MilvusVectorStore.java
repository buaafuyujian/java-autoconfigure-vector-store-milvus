package com.fyj.rag.vectorstore;

import java.util.List;

/**
 * Milvus 向量存储接口（Spring AI 风格）
 * <p>
 * 管理单个 Collection 的所有操作，包括分区管理、数据CRUD、向量搜索
 * <p>
 * 使用 Builder 模式构建请求，减少方法重载，提供更好的可读性和扩展性
 * <p>
 * 示例用法：
 * <pre>{@code
 * // 查询示例
 * QueryRequest queryRequest = QueryRequest.builder()
 *     .filterExpression("age > 18")
 *     .partitionName("user_partition")
 *     .offset(0)
 *     .limit(100)
 *     .build();
 * List<Document> docs = vectorStore.query(queryRequest);
 *
 * // 向量搜索示例
 * SearchRequest searchRequest = SearchRequest.builder()
 *     .vector(embeddingVector)
 *     .topK(10)
 *     .filter("category == 'tech'")
 *     .partitionNames(List.of("partition1"))
 *     .build();
 * List<SearchResult> results = vectorStore.similaritySearch(searchRequest);
 *
 * // 文本搜索示例（需要 EmbeddingModel）
 * SearchRequest textRequest = SearchRequest.query("什么是人工智能")
 *     .topK(5)
 *     .build();
 * List<SearchResult> results = vectorStore.similaritySearch(textRequest);
 * }</pre>
 */
public interface MilvusVectorStore {

    // ==================== Collection 信息 ====================

    /**
     * 获取 Collection 名称
     */
    String getCollectionName();

    /**
     * 获取 Collection 中的文档总数
     */
    long count();

    /**
     * 获取指定分区中的文档数量
     *
     * @param partitionName 分区名称
     */
    long count(String partitionName);

    // ==================== 分区管理 ====================

    /**
     * 创建分区
     *
     * @param partitionName 分区名称
     */
    void createPartition(String partitionName);

    /**
     * 删除分区
     *
     * @param partitionName 分区名称
     */
    void dropPartition(String partitionName);

    /**
     * 检查分区是否存在
     *
     * @param partitionName 分区名称
     */
    boolean hasPartition(String partitionName);

    /**
     * 列出所有分区
     */
    List<String> listPartitions();

    /**
     * 加载分区到内存
     *
     * @param partitionName 分区名称
     */
    void loadPartition(String partitionName);

    /**
     * 加载多个分区到内存
     *
     * @param partitionNames 分区名称列表
     */
    void loadPartitions(List<String> partitionNames);

    /**
     * 释放分区
     *
     * @param partitionName 分区名称
     */
    void releasePartition(String partitionName);

    // ==================== 数据操作 ====================

    /**
     * 添加文档到默认分区
     *
     * @param documents 文档列表
     */
    void add(List<? extends Document> documents);

    /**
     * 添加文档到指定分区
     *
     * @param documents 文档列表
     * @param partitionName 分区名称
     */
    void add(List<? extends Document> documents, String partitionName);

    /**
     * 根据 ID 列表删除文档
     *
     * @param ids ID 列表
     */
    void delete(List<String> ids);

    /**
     * 根据 ID 列表从指定分区删除文档
     *
     * @param ids ID 列表
     * @param partitionName 分区名称
     */
    void delete(List<String> ids, String partitionName);

    /**
     * 根据过滤表达式删除文档
     *
     * @param filterExpression 过滤表达式
     */
    void deleteByFilter(String filterExpression);

    /**
     * 根据过滤表达式从指定分区删除文档
     *
     * @param filterExpression 过滤表达式
     * @param partitionName 分区名称
     */
    void deleteByFilter(String filterExpression, String partitionName);

    /**
     * 更新或插入文档到默认分区
     *
     * @param documents 文档列表
     */
    void upsert(List<Document> documents);

    /**
     * 更新或插入文档到指定分区
     *
     * @param documents 文档列表
     * @param partitionName 分区名称
     */
    void upsert(List<Document> documents, String partitionName);

    // ==================== 根据 ID 获取 ====================

    /**
     * 根据 ID 列表获取文档
     *
     * @param ids ID 列表
     * @return 文档列表
     */
    List<Document> getById(List<String> ids);

    /**
     * 根据 ID 列表获取文档，返回指定类型
     *
     * @param ids ID 列表
     * @param clazz 文档类型（Document 的子类）
     * @return 文档列表
     */
    <T extends Document> List<T> getById(List<String> ids, Class<T> clazz);

    /**
     * 根据 ID 列表从指定分区获取文档
     *
     * @param ids ID 列表
     * @param partitionName 分区名称
     * @return 文档列表
     */
    List<Document> getById(List<String> ids, String partitionName);

    /**
     * 根据 ID 列表从指定分区获取文档，返回指定类型
     *
     * @param ids ID 列表
     * @param partitionName 分区名称
     * @param clazz 文档类型（Document 的子类）
     * @return 文档列表
     */
    <T extends Document> List<T> getById(List<String> ids, String partitionName, Class<T> clazz);

    // ==================== 查询操作（Spring AI 风格）====================

    /**
     * 使用 QueryRequest 查询文档
     * <p>
     * 这是推荐的查询方式，支持过滤、分区、分页等所有功能
     * <p>
     * 示例：
     * <pre>{@code
     * QueryRequest request = QueryRequest.builder()
     *     .filterExpression("category == 'tech'")
     *     .partitionName("articles")
     *     .offset(0)
     *     .limit(50)
     *     .build();
     * List<Document> docs = vectorStore.query(request);
     * }</pre>
     *
     * @param request 查询请求
     * @return 文档列表
     */
    List<Document> query(QueryRequest request);

    /**
     * 使用 QueryRequest 查询文档，返回指定类型
     *
     * @param request 查询请求
     * @param clazz 文档类型（Document 的子类）
     * @return 文档列表
     */
    <T extends Document> List<T> query(QueryRequest request, Class<T> clazz);

    /**
     * 简单查询：根据过滤表达式查询文档
     * <p>
     * 便捷方法，等同于 {@code query(QueryRequest.filter(filterExpression))}
     *
     * @param filterExpression 过滤表达式
     * @return 文档列表
     */
    default List<Document> query(String filterExpression) {
        return query(QueryRequest.filter(filterExpression));
    }

    /**
     * 简单查询：根据过滤表达式查询文档，返回指定类型
     *
     * @param filterExpression 过滤表达式
     * @param clazz 文档类型（Document 的子类）
     * @return 文档列表
     */
    default <T extends Document> List<T> query(String filterExpression, Class<T> clazz) {
        return query(QueryRequest.filter(filterExpression), clazz);
    }

    // ==================== 向量搜索（Spring AI 风格）====================

    /**
     * 向量相似度搜索
     * <p>
     * 这是推荐的搜索方式，支持向量/文本查询、过滤、分区等所有功能
     * <p>
     * 示例：
     * <pre>{@code
     * // 向量搜索
     * SearchRequest request = SearchRequest.builder()
     *     .vector(queryVector)
     *     .topK(10)
     *     .filter("category == 'tech'")
     *     .partitionNames(List.of("partition1", "partition2"))
     *     .similarityThreshold(0.7f)
     *     .build();
     * List<SearchResult> results = vectorStore.similaritySearch(request);
     *
     * // 文本搜索（需要 EmbeddingModel）
     * SearchRequest textRequest = SearchRequest.query("什么是机器学习")
     *     .topK(5)
     *     .build();
     * List<SearchResult> results = vectorStore.similaritySearch(textRequest);
     * }</pre>
     *
     * @param request 搜索请求
     * @return 搜索结果列表
     */
    List<SearchResult> similaritySearch(SearchRequest request);

    /**
     * 向量相似度搜索，返回指定类型
     *
     * @param request 搜索请求
     * @param clazz 文档类型（Document 的子类）
     * @return 搜索结果列表
     */
    <T extends Document> List<SearchResult<T>> similaritySearch(SearchRequest request, Class<T> clazz);


    // ==================== 数据管理 ====================

    /**
     * 刷新数据到磁盘
     */
    void flush();

    /**
     * 压缩 Collection
     */
    void compact();
}

