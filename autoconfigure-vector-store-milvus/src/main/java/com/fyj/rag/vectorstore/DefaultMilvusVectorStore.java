package com.fyj.rag.vectorstore;

import com.fyj.rag.exception.*;
import com.google.gson.Gson;
import com.google.gson.JsonObject;
import io.milvus.v2.client.MilvusClientV2;
import io.milvus.v2.service.partition.request.*;
import io.milvus.v2.service.utility.request.CompactReq;
import io.milvus.v2.service.utility.request.FlushReq;
import io.milvus.v2.service.vector.request.*;
import io.milvus.v2.service.vector.request.data.FloatVec;
import io.milvus.v2.service.vector.response.*;
import lombok.extern.slf4j.Slf4j;

import java.util.*;
import java.util.stream.Collectors;

/**
 * MilvusVectorStore 默认实现
 */
@Slf4j
public class DefaultMilvusVectorStore implements MilvusVectorStore {

    private final MilvusClientV2 client;
    private final String collectionName;
    private final String idFieldName;
    private final String contentFieldName;
    private final String embeddingFieldName;
    private final String metadataFieldName;
    private final List<String> outputFields;

    private static final Gson GSON = new Gson();

    public DefaultMilvusVectorStore(MilvusClientV2 client, String collectionName) {
        this(client, collectionName, "id", "content", "embedding", "metadata");
    }

    public DefaultMilvusVectorStore(MilvusClientV2 client, String collectionName,
                                    String idFieldName, String contentFieldName,
                                    String embeddingFieldName, String metadataFieldName) {
        this.client = client;
        this.collectionName = collectionName;
        this.idFieldName = idFieldName;
        this.contentFieldName = contentFieldName;
        this.embeddingFieldName = embeddingFieldName;
        this.metadataFieldName = metadataFieldName;
        this.outputFields = Arrays.asList(idFieldName, contentFieldName, metadataFieldName);
    }

    // ==================== Collection 信息 ====================

    @Override
    public String getCollectionName() {
        return collectionName;
    }

    @Override
    public long count() {
        try {
            QueryResp response = client.query(QueryReq.builder()
                    .collectionName(collectionName)
                    .filter("")
                    .outputFields(Collections.singletonList("count(*)"))
                    .build());
            if (response.getQueryResults() != null && !response.getQueryResults().isEmpty()) {
                Object count = response.getQueryResults().get(0).getEntity().get("count(*)");
                return count != null ? ((Number) count).longValue() : 0L;
            }
            return 0L;
        } catch (Exception e) {
            throw new MilvusDataException(ErrorCode.DATA_QUERY_FAILED,
                    "Failed to count documents in collection: " + collectionName, e);
        }
    }

    @Override
    public long count(String partitionName) {
        try {
            QueryResp response = client.query(QueryReq.builder()
                    .collectionName(collectionName)
                    .partitionNames(Collections.singletonList(partitionName))
                    .filter("")
                    .outputFields(Collections.singletonList("count(*)"))
                    .build());
            if (response.getQueryResults() != null && !response.getQueryResults().isEmpty()) {
                Object count = response.getQueryResults().get(0).getEntity().get("count(*)");
                return count != null ? ((Number) count).longValue() : 0L;
            }
            return 0L;
        } catch (Exception e) {
            throw new MilvusDataException(ErrorCode.DATA_QUERY_FAILED,
                    "Failed to count documents in partition: " + partitionName, e);
        }
    }

    // ==================== 分区管理 ====================

    @Override
    public void createPartition(String partitionName) {
        try {
            client.createPartition(CreatePartitionReq.builder()
                    .collectionName(collectionName)
                    .partitionName(partitionName)
                    .build());
            log.info("Created partition: {} in collection: {}", partitionName, collectionName);
        } catch (Exception e) {
            throw new MilvusPartitionException(ErrorCode.PARTITION_CREATE_FAILED,
                    "Failed to create partition: " + partitionName, e);
        }
    }

    @Override
    public void dropPartition(String partitionName) {
        try {
            client.dropPartition(DropPartitionReq.builder()
                    .collectionName(collectionName)
                    .partitionName(partitionName)
                    .build());
            log.info("Dropped partition: {} from collection: {}", partitionName, collectionName);
        } catch (Exception e) {
            throw new MilvusPartitionException(ErrorCode.PARTITION_DROP_FAILED,
                    "Failed to drop partition: " + partitionName, e);
        }
    }

    @Override
    public boolean hasPartition(String partitionName) {
        try {
            return client.hasPartition(HasPartitionReq.builder()
                    .collectionName(collectionName)
                    .partitionName(partitionName)
                    .build());
        } catch (Exception e) {
            throw new MilvusPartitionException(ErrorCode.PARTITION_NOT_FOUND,
                    "Failed to check partition: " + partitionName, e);
        }
    }

    @Override
    public List<String> listPartitions() {
        try {
            return client.listPartitions(ListPartitionsReq.builder()
                    .collectionName(collectionName)
                    .build());
        } catch (Exception e) {
            throw new MilvusPartitionException(ErrorCode.UNKNOWN_ERROR,
                    "Failed to list partitions in collection: " + collectionName, e);
        }
    }

    @Override
    public void loadPartition(String partitionName) {
        try {
            client.loadPartitions(LoadPartitionsReq.builder()
                    .collectionName(collectionName)
                    .partitionNames(Collections.singletonList(partitionName))
                    .build());
            log.info("Loaded partition: {} in collection: {}", partitionName, collectionName);
        } catch (Exception e) {
            throw new MilvusPartitionException(ErrorCode.PARTITION_LOAD_FAILED,
                    "Failed to load partition: " + partitionName, e);
        }
    }

    @Override
    public void loadPartitions(List<String> partitionNames) {
        try {
            client.loadPartitions(LoadPartitionsReq.builder()
                    .collectionName(collectionName)
                    .partitionNames(partitionNames)
                    .build());
            log.info("Loaded partitions: {} in collection: {}", partitionNames, collectionName);
        } catch (Exception e) {
            throw new MilvusPartitionException(ErrorCode.PARTITION_LOAD_FAILED,
                    "Failed to load partitions: " + partitionNames, e);
        }
    }

    @Override
    public void releasePartition(String partitionName) {
        try {
            client.releasePartitions(ReleasePartitionsReq.builder()
                    .collectionName(collectionName)
                    .partitionNames(Collections.singletonList(partitionName))
                    .build());
            log.info("Released partition: {} in collection: {}", partitionName, collectionName);
        } catch (Exception e) {
            throw new MilvusPartitionException(ErrorCode.PARTITION_RELEASE_FAILED,
                    "Failed to release partition: " + partitionName, e);
        }
    }

    // ==================== 数据操作 - 默认分区 ====================

    @Override
    public void add(List<Document> documents) {
        add(documents, null);
    }

    @Override
    public void delete(List<String> ids) {
        delete(ids, null);
    }

    @Override
    public void deleteByFilter(String filterExpression) {
        deleteByFilter(filterExpression, null);
    }

    // ==================== 数据操作 - 指定分区 ====================

    @Override
    public void add(List<Document> documents, String partitionName) {
        if (documents == null || documents.isEmpty()) {
            return;
        }
        try {
            List<JsonObject> data = documents.stream()
                    .map(this::documentToJsonObject)
                    .collect(Collectors.toList());

            InsertReq.InsertReqBuilder<?, ?> builder = InsertReq.builder()
                    .collectionName(collectionName)
                    .data(data);

            if (partitionName != null && !partitionName.isEmpty()) {
                builder.partitionName(partitionName);
            }

            InsertResp response = client.insert(builder.build());
            log.info("Inserted {} documents into collection: {}, partition: {}",
                    response.getInsertCnt(), collectionName, partitionName);
        } catch (Exception e) {
            throw new MilvusDataException(ErrorCode.DATA_INSERT_FAILED,
                    "Failed to insert documents into collection: " + collectionName, e);
        }
    }

    @Override
    public void delete(List<String> ids, String partitionName) {
        if (ids == null || ids.isEmpty()) {
            return;
        }
        try {
            DeleteReq.DeleteReqBuilder<?, ?> builder = DeleteReq.builder()
                    .collectionName(collectionName)
                    .ids(new ArrayList<>(ids));

            if (partitionName != null && !partitionName.isEmpty()) {
                builder.partitionName(partitionName);
            }

            DeleteResp response = client.delete(builder.build());
            log.info("Deleted {} documents from collection: {}, partition: {}",
                    response.getDeleteCnt(), collectionName, partitionName);
        } catch (Exception e) {
            throw new MilvusDataException(ErrorCode.DATA_DELETE_FAILED,
                    "Failed to delete documents by ids from collection: " + collectionName, e);
        }
    }

    @Override
    public void deleteByFilter(String filterExpression, String partitionName) {
        if (filterExpression == null || filterExpression.isEmpty()) {
            return;
        }
        try {
            DeleteReq.DeleteReqBuilder<?, ?> builder = DeleteReq.builder()
                    .collectionName(collectionName)
                    .filter(filterExpression);

            if (partitionName != null && !partitionName.isEmpty()) {
                builder.partitionName(partitionName);
            }

            DeleteResp response = client.delete(builder.build());
            log.info("Deleted {} documents by filter from collection: {}, partition: {}",
                    response.getDeleteCnt(), collectionName, partitionName);
        } catch (Exception e) {
            throw new MilvusDataException(ErrorCode.DATA_DELETE_FAILED,
                    "Failed to delete documents by filter from collection: " + collectionName, e);
        }
    }

    // ==================== Upsert 操作 ====================

    @Override
    public void upsert(List<Document> documents) {
        upsert(documents, null);
    }

    @Override
    public void upsert(List<Document> documents, String partitionName) {
        if (documents == null || documents.isEmpty()) {
            return;
        }
        try {
            List<JsonObject> data = documents.stream()
                    .map(this::documentToJsonObject)
                    .collect(Collectors.toList());

            UpsertReq.UpsertReqBuilder<?, ?> builder = UpsertReq.builder()
                    .collectionName(collectionName)
                    .data(data);

            if (partitionName != null && !partitionName.isEmpty()) {
                builder.partitionName(partitionName);
            }

            UpsertResp response = client.upsert(builder.build());
            log.info("Upserted {} documents into collection: {}, partition: {}",
                    response.getUpsertCnt(), collectionName, partitionName);
        } catch (Exception e) {
            throw new MilvusDataException(ErrorCode.DATA_UPSERT_FAILED,
                    "Failed to upsert documents into collection: " + collectionName, e);
        }
    }

    // ==================== 查询操作 ====================

    @Override
    public List<Document> getById(List<String> ids) {
        return getById(ids, null);
    }

    @Override
    public List<Document> getById(List<String> ids, String partitionName) {
        if (ids == null || ids.isEmpty()) {
            return Collections.emptyList();
        }
        try {
            GetReq.GetReqBuilder<?, ?> builder = GetReq.builder()
                    .collectionName(collectionName)
                    .ids(new ArrayList<>(ids))
                    .outputFields(outputFields);

            if (partitionName != null && !partitionName.isEmpty()) {
                builder.partitionName(partitionName);
            }

            GetResp response = client.get(builder.build());
            return response.getGetResults().stream()
                    .map(this::resultToDocument)
                    .collect(Collectors.toList());
        } catch (Exception e) {
            throw new MilvusDataException(ErrorCode.DATA_QUERY_FAILED,
                    "Failed to get documents by ids from collection: " + collectionName, e);
        }
    }

    @Override
    public List<Document> query(String filterExpression) {
        return query(filterExpression, null, 1000);
    }

    @Override
    public List<Document> query(String filterExpression, String partitionName) {
        return query(filterExpression, partitionName, 1000);
    }

    @Override
    public List<Document> query(String filterExpression, int limit) {
        return query(filterExpression, null, limit);
    }

    @Override
    public List<Document> query(String filterExpression, String partitionName, int limit) {
        try {
            QueryReq.QueryReqBuilder<?, ?> builder = QueryReq.builder()
                    .collectionName(collectionName)
                    .filter(filterExpression)
                    .outputFields(outputFields)
                    .limit(limit);

            if (partitionName != null && !partitionName.isEmpty()) {
                builder.partitionNames(Collections.singletonList(partitionName));
            }

            QueryResp response = client.query(builder.build());
            return response.getQueryResults().stream()
                    .map(this::queryResultToDocument)
                    .collect(Collectors.toList());
        } catch (Exception e) {
            throw new MilvusDataException(ErrorCode.DATA_QUERY_FAILED,
                    "Failed to query documents from collection: " + collectionName, e);
        }
    }

    // ==================== 向量搜索 - 全局 ====================

    @Override
    public List<SearchResult> similaritySearch(SearchRequest request) {
        return doSearch(request, null);
    }

    @Override
    public List<SearchResult> similaritySearch(List<Float> vector, int topK) {
        return similaritySearch(SearchRequest.of(vector, topK));
    }

    @Override
    public List<SearchResult> similaritySearch(List<Float> vector, int topK, String filter) {
        return similaritySearch(SearchRequest.of(vector, topK, filter));
    }

    // ==================== 向量搜索 - 指定分区 ====================

    @Override
    public List<SearchResult> similaritySearchInPartition(SearchRequest request, String partitionName) {
        return doSearch(request, Collections.singletonList(partitionName));
    }

    @Override
    public List<SearchResult> similaritySearchInPartition(List<Float> vector, int topK, String partitionName) {
        return similaritySearchInPartition(SearchRequest.of(vector, topK), partitionName);
    }

    @Override
    public List<SearchResult> similaritySearchInPartitions(SearchRequest request, List<String> partitionNames) {
        return doSearch(request, partitionNames);
    }

    @Override
    public List<SearchResult> similaritySearchInPartitions(List<Float> vector, int topK, List<String> partitionNames) {
        return similaritySearchInPartitions(SearchRequest.of(vector, topK), partitionNames);
    }

    private List<SearchResult> doSearch(SearchRequest request, List<String> partitionNames) {
        try {
            SearchReq.SearchReqBuilder<?, ?> builder = SearchReq.builder()
                    .collectionName(collectionName)
                    .annsField(request.getVectorFieldName())
                    .data(Collections.singletonList(new FloatVec(request.getVector())))
                    .topK(request.getTopK())
                    .outputFields(outputFields);

            if (request.getFilter() != null && !request.getFilter().isEmpty()) {
                builder.filter(request.getFilter());
            }

            if (request.getOffset() > 0) {
                builder.offset(request.getOffset());
            }

            if (request.getSearchParams() != null && !request.getSearchParams().isEmpty()) {
                builder.searchParams(request.getSearchParams());
            }

            if (partitionNames != null && !partitionNames.isEmpty()) {
                builder.partitionNames(partitionNames);
            }

            SearchResp response = client.search(builder.build());

            List<SearchResult> results = new ArrayList<>();
            for (List<SearchResp.SearchResult> searchResults : response.getSearchResults()) {
                for (SearchResp.SearchResult result : searchResults) {
                    Document doc = searchResultToDocument(result);
                    float score = result.getScore();

                    // 过滤相似度阈值
                    if (score >= request.getSimilarityThreshold()) {
                        results.add(SearchResult.of(doc, score, score));
                    }
                }
            }
            return results;
        } catch (Exception e) {
            throw new MilvusSearchException(ErrorCode.SEARCH_FAILED,
                    "Failed to search in collection: " + collectionName, e);
        }
    }

    // ==================== 数据管理 ====================

    @Override
    public void flush() {
        try {
            client.flush(FlushReq.builder()
                                 .collectionNames(Collections.singletonList(collectionName))
                                 .build());
            log.info("Flushed collection: {}", collectionName);
        } catch (Exception e) {
            throw new MilvusDataException(ErrorCode.UNKNOWN_ERROR,
                    "Failed to flush collection: " + collectionName, e);
        }
    }

    @Override
    public void compact() {
        try {
            client.compact(CompactReq.builder()
                                     .collectionName(collectionName)
                                     .build());
            log.info("Compacted collection: {}", collectionName);
        } catch (Exception e) {
            throw new MilvusDataException(ErrorCode.UNKNOWN_ERROR,
                    "Failed to compact collection: " + collectionName, e);
        }
    }

    // ==================== 辅助方法 ====================

    private JsonObject documentToJsonObject(Document document) {
        JsonObject json = new JsonObject();
        json.addProperty(idFieldName, document.getId());

        if (document.getContent() != null) {
            json.addProperty(contentFieldName, document.getContent());
        }

        if (document.getEmbedding() != null) {
            json.add(embeddingFieldName, GSON.toJsonTree(document.getEmbedding()));
        }

        if (document.getMetadata() != null && !document.getMetadata().isEmpty()) {
            json.add(metadataFieldName, GSON.toJsonTree(document.getMetadata()));
        }

        return json;
    }

    private Document resultToDocument(QueryResp.QueryResult result) {
        Map<String, Object> entity = result.getEntity();
        return entityToDocument(entity);
    }

    private Document queryResultToDocument(QueryResp.QueryResult result) {
        Map<String, Object> entity = result.getEntity();
        return entityToDocument(entity);
    }

    private Document searchResultToDocument(SearchResp.SearchResult result) {
        Map<String, Object> entity = result.getEntity();
        return entityToDocument(entity);
    }

    @SuppressWarnings("unchecked")
    private Document entityToDocument(Map<String, Object> entity) {
        Document.DocumentBuilder builder = Document.builder();

        Object id = entity.get(idFieldName);
        if (id != null) {
            builder.id(id.toString());
        }

        Object content = entity.get(contentFieldName);
        if (content != null) {
            builder.content(content.toString());
        }

        Object embedding = entity.get(embeddingFieldName);
        if (embedding instanceof List) {
            builder.embedding((List<Float>) embedding);
        }

        Object metadata = entity.get(metadataFieldName);
        if (metadata instanceof Map) {
            builder.metadata((Map<String, Object>) metadata);
        } else if (metadata instanceof String) {
            try {
                Map<String, Object> metadataMap = GSON.fromJson((String) metadata, Map.class);
                builder.metadata(metadataMap);
            } catch (Exception ignored) {
                // ignore json parse error
            }
        }

        return builder.build();
    }
}

