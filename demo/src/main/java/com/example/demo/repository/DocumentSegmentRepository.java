package com.example.demo.repository;

import com.example.demo.entity.DocumentSegment;
import com.fyj.rag.client.MilvusClient;
import com.fyj.rag.vectorstore.Document;
import com.fyj.rag.vectorstore.MilvusVectorStore;
import com.fyj.rag.vectorstore.SearchRequest;
import com.fyj.rag.vectorstore.SearchResult;
import com.google.gson.Gson;
import com.google.gson.JsonObject;
import io.milvus.v2.client.MilvusClientV2;
import io.milvus.v2.service.vector.request.InsertReq;
import io.milvus.v2.service.vector.request.QueryReq;
import io.milvus.v2.service.vector.request.SearchReq;
import io.milvus.v2.service.vector.request.UpsertReq;
import io.milvus.v2.service.vector.request.data.FloatVec;
import io.milvus.v2.service.vector.response.QueryResp;
import io.milvus.v2.service.vector.response.SearchResp;
import lombok.extern.slf4j.Slf4j;

import java.util.*;
import java.util.stream.Collectors;

/**
 * 文档片段 Repository
 * <p>
 * 提供 DocumentSegment 的完整 CRUD 操作
 */
@Slf4j
public class DocumentSegmentRepository {

    private final MilvusClient milvusClient;
    private final MilvusClientV2 nativeClient;
    private final MilvusVectorStore vectorStore;
    private final int dimension;

    private static final Gson GSON = new Gson();
    private static final List<String> OUTPUT_FIELDS = Arrays.asList(
            DocumentSegment.FIELD_ID,
            DocumentSegment.FIELD_KNOWLEDGE_ID,
            DocumentSegment.FIELD_FILE_ID,
            DocumentSegment.FIELD_CONTENT,
            DocumentSegment.FIELD_METADATA
    );

    public DocumentSegmentRepository(MilvusClient milvusClient) {
        this(milvusClient, DocumentSegment.DEFAULT_DIMENSION);
    }

    public DocumentSegmentRepository(MilvusClient milvusClient, int dimension) {
        this.milvusClient = milvusClient;
        this.nativeClient = milvusClient.getNativeClient();
        this.dimension = dimension;
        this.vectorStore = milvusClient.getVectorStore(DocumentSegment.COLLECTION_NAME);
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
     */
    public void insert(List<DocumentSegment> segments) {
        if (segments == null || segments.isEmpty()) {
            return;
        }

        List<JsonObject> data = segments.stream()
                .map(this::toJsonObject)
                .collect(Collectors.toList());

        nativeClient.insert(InsertReq.builder()
                .collectionName(DocumentSegment.COLLECTION_NAME)
                .data(data)
                .build());

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

        List<JsonObject> data = segments.stream()
                .map(this::toJsonObject)
                .collect(Collectors.toList());

        nativeClient.upsert(UpsertReq.builder()
                .collectionName(DocumentSegment.COLLECTION_NAME)
                .data(data)
                .build());

        log.info("Upserted {} document segments", segments.size());
    }

    // ==================== 查询操作 ====================

    /**
     * 根据 ID 获取文档片段
     */
    public Optional<DocumentSegment> findById(String id) {
        List<DocumentSegment> results = findByIds(Collections.singletonList(id));
        return results.isEmpty() ? Optional.empty() : Optional.of(results.get(0));
    }

    /**
     * 根据 ID 列表获取文档片段
     */
    public List<DocumentSegment> findByIds(List<String> ids) {
        if (ids == null || ids.isEmpty()) {
            return Collections.emptyList();
        }

        String filter = String.format("%s in [\"%s\"]",
                DocumentSegment.FIELD_ID,
                String.join("\", \"", ids));

        return query(filter, ids.size());
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
        return query(DocumentSegment.filterByKnowledgeId(knowledgeId), limit);
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
        return query(DocumentSegment.filterByFileId(fileId), limit);
    }

    /**
     * 根据知识库ID和文档ID查询片段
     */
    public List<DocumentSegment> findByKnowledgeIdAndFileId(String knowledgeId, String fileId) {
        return query(DocumentSegment.filterByKnowledgeIdAndFileId(knowledgeId, fileId), 10000);
    }

    /**
     * 通用查询方法
     */
    public List<DocumentSegment> query(String filter, int limit) {
        QueryResp response = nativeClient.query(QueryReq.builder()
                .collectionName(DocumentSegment.COLLECTION_NAME)
                .filter(filter)
                .outputFields(OUTPUT_FIELDS)
                .limit(limit)
                .build());

        return response.getQueryResults().stream()
                .map(this::fromQueryResult)
                .collect(Collectors.toList());
    }

    /**
     * 统计知识库中的片段数量
     */
    public long countByKnowledgeId(String knowledgeId) {
        QueryResp response = nativeClient.query(QueryReq.builder()
                .collectionName(DocumentSegment.COLLECTION_NAME)
                .filter(DocumentSegment.filterByKnowledgeId(knowledgeId))
                .outputFields(Collections.singletonList("count(*)"))
                .build());

        if (response.getQueryResults() != null && !response.getQueryResults().isEmpty()) {
            Object count = response.getQueryResults().get(0).getEntity().get("count(*)");
            return count != null ? ((Number) count).longValue() : 0L;
        }
        return 0L;
    }

    /**
     * 统计文档中的片段数量
     */
    public long countByFileId(String fileId) {
        QueryResp response = nativeClient.query(QueryReq.builder()
                .collectionName(DocumentSegment.COLLECTION_NAME)
                .filter(DocumentSegment.filterByFileId(fileId))
                .outputFields(Collections.singletonList("count(*)"))
                .build());

        if (response.getQueryResults() != null && !response.getQueryResults().isEmpty()) {
            Object count = response.getQueryResults().get(0).getEntity().get("count(*)");
            return count != null ? ((Number) count).longValue() : 0L;
        }
        return 0L;
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
        SearchReq.SearchReqBuilder<?, ?> builder = SearchReq.builder()
                .collectionName(DocumentSegment.COLLECTION_NAME)
                .annsField(DocumentSegment.FIELD_EMBEDDING)
                .data(Collections.singletonList(new FloatVec(vector)))
                .topK(topK)
                .outputFields(OUTPUT_FIELDS);

        if (filter != null && !filter.isEmpty()) {
            builder.filter(filter);
        }

        SearchResp response = nativeClient.search(builder.build());

        List<DocumentSegmentSearchResult> results = new ArrayList<>();
        for (List<SearchResp.SearchResult> searchResults : response.getSearchResults()) {
            for (SearchResp.SearchResult result : searchResults) {
                DocumentSegment segment = fromSearchResult(result);
                results.add(new DocumentSegmentSearchResult(segment, result.getScore()));
            }
        }
        return results;
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

    // ==================== 辅助方法 ====================

    private JsonObject toJsonObject(DocumentSegment segment) {
        JsonObject json = new JsonObject();
        json.addProperty(DocumentSegment.FIELD_ID, segment.getId());
        json.addProperty(DocumentSegment.FIELD_KNOWLEDGE_ID, segment.getKnowledgeId());
        json.addProperty(DocumentSegment.FIELD_FILE_ID, segment.getFileId());
        json.addProperty(DocumentSegment.FIELD_CONTENT, segment.getContent());
        json.add(DocumentSegment.FIELD_EMBEDDING, GSON.toJsonTree(segment.getEmbedding()));
        json.add(DocumentSegment.FIELD_METADATA, GSON.toJsonTree(segment.getMetadata()));
        return json;
    }

    @SuppressWarnings("unchecked")
    private DocumentSegment fromQueryResult(QueryResp.QueryResult result) {
        Map<String, Object> entity = result.getEntity();
        return fromEntity(entity);
    }

    @SuppressWarnings("unchecked")
    private DocumentSegment fromSearchResult(SearchResp.SearchResult result) {
        Map<String, Object> entity = result.getEntity();
        return fromEntity(entity);
    }

    @SuppressWarnings("unchecked")
    private DocumentSegment fromEntity(Map<String, Object> entity) {
        DocumentSegment.DocumentSegmentBuilder builder = DocumentSegment.builder();

        Object id = entity.get(DocumentSegment.FIELD_ID);
        if (id != null) {
            builder.id(id.toString());
        }

        Object knowledgeId = entity.get(DocumentSegment.FIELD_KNOWLEDGE_ID);
        if (knowledgeId != null) {
            builder.knowledgeId(knowledgeId.toString());
        }

        Object fileId = entity.get(DocumentSegment.FIELD_FILE_ID);
        if (fileId != null) {
            builder.fileId(fileId.toString());
        }

        Object content = entity.get(DocumentSegment.FIELD_CONTENT);
        if (content != null) {
            builder.content(content.toString());
        }

        Object embedding = entity.get(DocumentSegment.FIELD_EMBEDDING);
        if (embedding instanceof List) {
            builder.embedding((List<Float>) embedding);
        }

        Object metadata = entity.get(DocumentSegment.FIELD_METADATA);
        if (metadata instanceof Map) {
            builder.metadata((Map<String, Object>) metadata);
        } else if (metadata instanceof String) {
            try {
                Map<String, Object> metadataMap = GSON.fromJson((String) metadata, Map.class);
                builder.metadata(metadataMap);
            } catch (Exception ignored) {
            }
        }

        return builder.build();
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

