package com.fyj.rag.vectorstore;

import java.util.List;

/**
 * Milvus 向量存储接口
 * <p>
 * 管理单个 Collection 的所有操作，包括分区管理、数据CRUD、向量搜索
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
     */
    long count(String partitionName);

    // ==================== 分区管理 ====================

    /**
     * 创建分区
     */
    void createPartition(String partitionName);

    /**
     * 删除分区
     */
    void dropPartition(String partitionName);

    /**
     * 检查分区是否存在
     */
    boolean hasPartition(String partitionName);

    /**
     * 列出所有分区
     */
    List<String> listPartitions();

    /**
     * 加载分区到内存
     */
    void loadPartition(String partitionName);

    /**
     * 加载多个分区到内存
     */
    void loadPartitions(List<String> partitionNames);

    /**
     * 释放分区
     */
    void releasePartition(String partitionName);

    // ==================== 数据操作 - 默认分区 ====================

    /**
     * 添加文档到默认分区
     */
    void add(List<Document> documents);

    /**
     * 根据 ID 列表删除文档
     */
    void delete(List<String> ids);

    /**
     * 根据过滤表达式删除文档
     */
    void deleteByFilter(String filterExpression);

    // ==================== 数据操作 - 指定分区 ====================

    /**
     * 添加文档到指定分区
     */
    void add(List<Document> documents, String partitionName);

    /**
     * 根据 ID 列表从指定分区删除文档
     */
    void delete(List<String> ids, String partitionName);

    /**
     * 根据过滤表达式从指定分区删除文档
     */
    void deleteByFilter(String filterExpression, String partitionName);

    // ==================== Upsert 操作 ====================

    /**
     * 更新或插入文档到默认分区
     */
    void upsert(List<Document> documents);

    /**
     * 更新或插入文档到指定分区
     */
    void upsert(List<Document> documents, String partitionName);

    // ==================== 查询操作 ====================

    /**
     * 根据 ID 列表获取文档
     */
    List<Document> getById(List<String> ids);

    /**
     * 根据 ID 列表从指定分区获取文档
     */
    List<Document> getById(List<String> ids, String partitionName);

    /**
     * 根据过滤表达式查询文档
     */
    List<Document> query(String filterExpression);

    /**
     * 根据过滤表达式从指定分区查询文档
     */
    List<Document> query(String filterExpression, String partitionName);

    /**
     * 根据过滤表达式查询文档（带分页）
     *
     * @param filterExpression 过滤表达式
     * @param offset 偏移量
     * @param limit 限制数量
     */
    List<Document> query(String filterExpression, int offset, int limit);

    /**
     * 根据过滤表达式从指定分区查询文档（带分页）
     *
     * @param filterExpression 过滤表达式
     * @param partitionName 分区名
     * @param offset 偏移量
     * @param limit 限制数量
     */
    List<Document> query(String filterExpression, String partitionName, int offset, int limit);

    // ==================== 向量搜索 - 全局 ====================

    /**
     * 向量相似度搜索
     */
    List<SearchResult> similaritySearch(SearchRequest request);

    /**
     * 向量相似度搜索（简化版）
     */
    List<SearchResult> similaritySearch(List<Float> vector, int topK);

    /**
     * 向量相似度搜索（带过滤条件）
     */
    List<SearchResult> similaritySearch(List<Float> vector, int topK, String filter);

    // ==================== 向量搜索 - 指定分区 ====================

    /**
     * 在指定分区进行向量相似度搜索
     */
    List<SearchResult> similaritySearchInPartition(SearchRequest request, String partitionName);

    /**
     * 在指定分区进行向量相似度搜索（简化版）
     */
    List<SearchResult> similaritySearchInPartition(List<Float> vector, int topK, String partitionName);

    /**
     * 在多个分区进行向量相似度搜索
     */
    List<SearchResult> similaritySearchInPartitions(SearchRequest request, List<String> partitionNames);

    /**
     * 在多个分区进行向量相似度搜索（简化版）
     */
    List<SearchResult> similaritySearchInPartitions(List<Float> vector, int topK, List<String> partitionNames);

    // ==================== 文本搜索（需要 EmbeddingModel）====================

    /**
     * 使用文本进行相似度搜索（自动转换为向量）
     * <p>
     * 需要在创建 VectorStore 时提供 EmbeddingModel
     *
     * @param query 查询文本
     * @param topK 返回数量
     * @return 搜索结果
     */
    List<SearchResult> similaritySearch(String query, int topK);

    /**
     * 使用文本进行相似度搜索（带过滤条件）
     */
    List<SearchResult> similaritySearch(String query, int topK, String filter);

    /**
     * 在指定分区使用文本进行相似度搜索
     */
    List<SearchResult> similaritySearchInPartition(String query, int topK, String partitionName);

    /**
     * 在多个分区使用文本进行相似度搜索
     */
    List<SearchResult> similaritySearchInPartitions(String query, int topK, List<String> partitionNames);

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

