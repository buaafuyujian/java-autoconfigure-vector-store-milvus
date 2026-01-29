package com.example.demo.repository;

import com.example.demo.entity.DocumentSegment;
import com.fyj.rag.client.MilvusClient;
import com.fyj.rag.vectorstore.Document;
import com.fyj.rag.vectorstore.MilvusVectorStore;
import com.fyj.rag.vectorstore.SearchRequest;
import com.fyj.rag.vectorstore.SearchResult;
import lombok.extern.slf4j.Slf4j;

import java.util.*;
import java.util.stream.Collectors;

/**
 * 文档片段 Repository
 * <p>
 * 基于 MilvusVectorStore 提供 DocumentSegment 的 CRUD 操作
 * DocumentSegment 继承自 Document，可直接使用 MilvusVectorStore 的方法
 */
@Slf4j
public class DocumentSegmentRepository {

    private final MilvusClient milvusClient;
    private final MilvusVectorStore vectorStore;
    private final int dimension;

    public DocumentSegmentRepository(MilvusClient milvusClient) {
        this(milvusClient, DocumentSegment.DEFAULT_DIMENSION);
    }

    public DocumentSegmentRepository(MilvusClient milvusClient, int dimension) {
        this.milvusClient = milvusClient;
        this.dimension = dimension;
        // 使用带额外输出字段的 VectorStore，确保查询时能获取 knowledge_id 和 file_id
        this.vectorStore = milvusClient.getVectorStore(
                DocumentSegment.COLLECTION_NAME,
                DocumentSegment.FIELD_ID,
                DocumentSegment.FIELD_CONTENT,
                DocumentSegment.FIELD_EMBEDDING,
                DocumentSegment.FIELD_METADATA,
                Arrays.asList(DocumentSegment.FIELD_KNOWLEDGE_ID, DocumentSegment.FIELD_FILE_ID)
        );
    }

    // ==================== Collection 管理 ====================

    /**
     * 初始化 Collection（如果不存在则创建）
     */
    public void initCollection() {
        if (!milvusClient.hasCollection(DocumentSegment.COLLECTION_NAME)) {
            log.info("Creating collection: {}", DocumentSegment.COLLECTION_NAME);
            milvusClient.createCollection(
                    DocumentSegment.COLLECTION_NAME,
                    DocumentSegment.createSchema(dimension),
                    DocumentSegment.createIndex()
            );
            milvusClient.loadCollection(DocumentSegment.COLLECTION_NAME);
            log.info("Collection created and loaded: {}", DocumentSegment.COLLECTION_NAME);
        } else {
            log.info("Collection already exists: {}", DocumentSegment.COLLECTION_NAME);
        }
    }

    /**
     * 删除 Collection
     */
    public void dropCollection() {
        if (milvusClient.hasCollection(DocumentSegment.COLLECTION_NAME)) {
            milvusClient.releaseCollection(DocumentSegment.COLLECTION_NAME);
            milvusClient.dropCollection(DocumentSegment.COLLECTION_NAME);
            log.info("Collection dropped: {}", DocumentSegment.COLLECTION_NAME);
        }
    }

    // ==================== 插入操作 ====================

    /**
     * 插入单个文档片段
     */
    public void insert(DocumentSegment segment) {
        insert(Collections.singletonList(segment));
    }

    /**
     * 批量插入文档片段
     * DocumentSegment 继承自 Document，可直接传入 vectorStore.add()
     */
    public void insert(List<DocumentSegment> segments) {
        if (segments == null || segments.isEmpty()) {
            return;
        }
        // DocumentSegment 是 Document 的子类，直接转换
        List<Document> documents = new ArrayList<>(segments);
        vectorStore.add(documents);
        log.info("Inserted {} document segments", segments.size());
    }

    /**
     * 插入或更新文档片段
     */
    public void upsert(DocumentSegment segment) {
        upsert(Collections.singletonList(segment));
    }

    /**
     * 批量插入或更新文档片段
     */
    public void upsert(List<DocumentSegment> segments) {
        if (segments == null || segments.isEmpty()) {
            return;
        }
        List<Document> documents = new ArrayList<>(segments);
        vectorStore.upsert(documents);
        log.info("Upserted {} document segments", segments.size());
    }

    // ==================== 查询操作 ====================

    /**
     * 根据 ID 获取文档片段
     */
    public Optional<DocumentSegment> findById(String id) {
        List<Document> results = vectorStore.getById(Collections.singletonList(id));
        if (results.isEmpty()) {
            return Optional.empty();
        }
        return Optional.of(DocumentSegment.fromDocument(results.get(0)));
    }

    /**
     * 根据 ID 列表获取文档片段
     */
    public List<DocumentSegment> findByIds(List<String> ids) {
        if (ids == null || ids.isEmpty()) {
            return Collections.emptyList();
        }
        List<Document> results = vectorStore.getById(ids);
        return results.stream()
                .map(DocumentSegment::fromDocument)
                .collect(Collectors.toList());
    }

    /**
     * 根据知识库ID查询所有片段
     */
    public List<DocumentSegment> findByKnowledgeId(String knowledgeId) {
        return findByKnowledgeId(knowledgeId, 10000);
    }

    /**
     * 根据知识库ID查询片段（带限制）
     */
    public List<DocumentSegment> findByKnowledgeId(String knowledgeId, int limit) {
        List<Document> results = vectorStore.query(DocumentSegment.filterByKnowledgeId(knowledgeId), limit);
        return results.stream()
                .map(DocumentSegment::fromDocument)
                .collect(Collectors.toList());
    }

    /**
     * 根据文档ID查询所有片段
     */
    public List<DocumentSegment> findByFileId(String fileId) {
        return findByFileId(fileId, 10000);
    }

    /**
     * 根据文档ID查询片段（带限制）
     */
    public List<DocumentSegment> findByFileId(String fileId, int limit) {
        List<Document> results = vectorStore.query(DocumentSegment.filterByFileId(fileId), limit);
        return results.stream()
                .map(DocumentSegment::fromDocument)
                .collect(Collectors.toList());
    }

    /**
     * 根据知识库ID和文档ID查询片段
     */
    public List<DocumentSegment> findByKnowledgeIdAndFileId(String knowledgeId, String fileId) {
        List<Document> results = vectorStore.query(
                DocumentSegment.filterByKnowledgeIdAndFileId(knowledgeId, fileId), 10000);
        return results.stream()
                .map(DocumentSegment::fromDocument)
                .collect(Collectors.toList());
    }

    /**
     * 统计知识库中的片段数量
     */
    public long countByKnowledgeId(String knowledgeId) {
        List<Document> results = vectorStore.query(DocumentSegment.filterByKnowledgeId(knowledgeId), 100000);
        return results.size();
    }

    /**
     * 统计文档中的片段数量
     */
    public long countByFileId(String fileId) {
        List<Document> results = vectorStore.query(DocumentSegment.filterByFileId(fileId), 100000);
        return results.size();
    }

    // ==================== 向量搜索 ====================

    /**
     * 在整个 Collection 中搜索相似片段
     */
    public List<DocumentSegmentSearchResult> search(List<Float> vector, int topK) {
        return search(vector, topK, null);
    }

    /**
     * 在指定知识库中搜索相似片段
     */
    public List<DocumentSegmentSearchResult> searchInKnowledge(List<Float> vector, int topK, String knowledgeId) {
        return search(vector, topK, DocumentSegment.filterByKnowledgeId(knowledgeId));
    }

    /**
     * 在指定文档中搜索相似片段
     */
    public List<DocumentSegmentSearchResult> searchInFile(List<Float> vector, int topK, String fileId) {
        return search(vector, topK, DocumentSegment.filterByFileId(fileId));
    }

    /**
     * 在多个知识库中搜索相似片段
     */
    public List<DocumentSegmentSearchResult> searchInKnowledges(List<Float> vector, int topK, List<String> knowledgeIds) {
        String filter = String.format("%s in [\"%s\"]",
                DocumentSegment.FIELD_KNOWLEDGE_ID,
                String.join("\", \"", knowledgeIds));
        return search(vector, topK, filter);
    }

    /**
     * 通用搜索方法
     */
    public List<DocumentSegmentSearchResult> search(List<Float> vector, int topK, String filter) {
        List<SearchResult> results;
        if (filter != null && !filter.isEmpty()) {
            results = vectorStore.similaritySearch(vector, topK, filter);
        } else {
            results = vectorStore.similaritySearch(vector, topK);
        }

        return results.stream()
                .map(r -> new DocumentSegmentSearchResult(
                        DocumentSegment.fromDocument(r.getDocument()),
                        r.getScore()))
                .collect(Collectors.toList());
    }

    // ==================== 删除操作 ====================

    /**
     * 根据 ID 删除文档片段
     */
    public void deleteById(String id) {
        deleteByIds(Collections.singletonList(id));
    }

    /**
     * 根据 ID 列表删除文档片段
     */
    public void deleteByIds(List<String> ids) {
        if (ids == null || ids.isEmpty()) {
            return;
        }
        vectorStore.delete(ids);
        log.info("Deleted {} document segments by ids", ids.size());
    }

    /**
     * 删除知识库下的所有片段
     */
    public void deleteByKnowledgeId(String knowledgeId) {
        vectorStore.deleteByFilter(DocumentSegment.filterByKnowledgeId(knowledgeId));
        log.info("Deleted all segments in knowledge: {}", knowledgeId);
    }

    /**
     * 删除文档下的所有片段
     */
    public void deleteByFileId(String fileId) {
        vectorStore.deleteByFilter(DocumentSegment.filterByFileId(fileId));
        log.info("Deleted all segments in file: {}", fileId);
    }

    /**
     * 根据过滤条件删除
     */
    public void deleteByFilter(String filter) {
        vectorStore.deleteByFilter(filter);
        log.info("Deleted segments by filter: {}", filter);
    }

    // ==================== 获取底层 VectorStore ====================

    /**
     * 获取底层的 MilvusVectorStore，用于更复杂的操作
     */
    public MilvusVectorStore getVectorStore() {
        return vectorStore;
    }

    /**
     * 搜索结果封装
     */
    @lombok.Data
    @lombok.AllArgsConstructor
    public static class DocumentSegmentSearchResult {
        private DocumentSegment segment;
        private float score;
    }
}

