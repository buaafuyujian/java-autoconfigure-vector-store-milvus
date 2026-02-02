package com.fyj.rag.vectorstore;

import com.fyj.rag.vectorstore.model.Document;
import com.fyj.rag.vectorstore.model.SearchResult;
import com.fyj.rag.vectorstore.request.QueryRequest;
import com.fyj.rag.vectorstore.request.SearchRequest;

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
 * List<SearchResult> results = vectorStore.search(searchRequest);
 *
 * // 文本搜索示例（需要 EmbeddingModel）
 * SearchRequest textRequest = SearchRequest.query("什么是人工智能")
 *     .topK(5)
 *     .build();
 * List<SearchResult> results = vectorStore.search(textRequest);
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
     * 使用 QueryRequest 查询文档（泛型版本）
     * <p>
     * 这是推荐的查询方式，支持过滤、分区、分页、指定返回类型等所有功能
     * <p>
     * 示例：
     * <pre>{@code
     * // 指定返回类型
     * QueryRequest<FaqDocument> request = QueryRequest.<FaqDocument>builder()
     *     .filter("category == 'tech'")
     *     .inPartition("articles")
     *     .limit(50)
     *     .as(FaqDocument.class)
     *     .build();
     * List<FaqDocument> docs = vectorStore.query(request);
     *
     * // 默认返回 Document 类型
     * QueryRequest<Document> request = QueryRequest.filter("type == 'faq'").build();
     * List<Document> docs = vectorStore.query(request);
     * }</pre>
     *
     * @param request 查询请求（包含返回类型信息）
     * @return 文档列表
     */
    <T extends Document> List<T> query(QueryRequest<T> request);

    /**
     * 简单查询：根据过滤表达式查询文档
     * <p>
     * 便捷方法，返回 Document 类型
     *
     * @param filterExpression 过滤表达式
     * @return 文档列表
     */
    default List<Document> query(String filterExpression) {
        return query(QueryRequest.of(filterExpression));
    }

    /**
     * 简单查询：根据过滤表达式查询文档，返回指定类型
     * <p>
     * 便捷方法
     *
     * @param filterExpression 过滤表达式
     * @param clazz 文档类型
     * @return 文档列表
     */
    default <T extends Document> List<T> query(String filterExpression, Class<T> clazz) {
        return query(QueryRequest.<T>builder().filter(filterExpression).documentClass(clazz).build());
    }

    // ==================== 搜索操作（Spring AI 风格）====================

    /**
     * 搜索文档（支持向量搜索、BM25搜索、混合搜索）
     * <p>
     * 这是推荐的搜索方式，支持向量/文本查询、过滤、分区、指定返回类型等所有功能
     * <p>
     * 支持三种搜索类型：
     * <ul>
     *     <li>{@link com.fyj.rag.vectorstore.request.SearchType#VECTOR} - 向量相似度搜索（默认）</li>
     *     <li>{@link com.fyj.rag.vectorstore.request.SearchType#BM25} - BM25 全文检索</li>
     *     <li>{@link com.fyj.rag.vectorstore.request.SearchType#HYBRID} - 混合搜索（向量 + BM25）</li>
     * </ul>
     * <p>
     * 示例：
     * <pre>{@code
     * // 1. 文本搜索（向量相似度），指定返回类型
     * SearchRequest<FaqDocument> request = SearchRequest.<FaqDocument>builder()
     *     .query("RAG 是什么")
     *     .topK(10)
     *     .filter("type == 'faq'")
     *     .inPartition("knowledge_base")
     *     .documentClass(FaqDocument.class)
     *     .build();
     * List<SearchResult<FaqDocument>> results = vectorStore.search(request);
     *
     * // 2. 向量搜索
     * SearchRequest<Document> request = SearchRequest.<Document>builder()
     *     .vector(queryVector)
     *     .topK(10)
     *     .similarityThreshold(0.7f)
     *     .build();
     * List<SearchResult<Document>> results = vectorStore.search(request);
     *
     * // 3. BM25 全文检索
     * SearchRequest<Document> bm25Request = SearchRequest.<Document>builder()
     *     .query("人工智能 机器学习")
     *     .searchType(SearchType.BM25)
     *     .textFieldName("content")
     *     .topK(10)
     *     .build();
     * List<SearchResult<Document>> results = vectorStore.search(bm25Request);
     *
     * // 或使用便捷方法
     * SearchRequest<Document> bm25Request = SearchRequest.bm25("人工智能", 10);
     *
     * // 4. 混合搜索（向量 + BM25）
     * SearchRequest<Document> hybridRequest = SearchRequest.<Document>builder()
     *     .query("什么是深度学习")
     *     .searchType(SearchType.HYBRID)
     *     .vectorWeight(0.7f)
     *     .bm25Weight(0.3f)
     *     .topK(10)
     *     .build();
     * List<SearchResult<Document>> results = vectorStore.search(hybridRequest);
     *
     * // 或使用便捷方法
     * SearchRequest<Document> hybridRequest = SearchRequest.hybrid("深度学习", 10, 0.7f, 0.3f);
     * }</pre>
     *
     * @param request 搜索请求（包含搜索类型和返回类型信息）
     * @return 搜索结果列表
     */
    <T extends Document> List<SearchResult<T>> search(SearchRequest<T> request);


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

