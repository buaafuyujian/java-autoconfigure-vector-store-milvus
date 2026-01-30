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
import org.springframework.ai.embedding.EmbeddingModel;

import java.util.*;
import java.util.stream.Collectors;

/**
 * MilvusVectorStore 默认实现
 * <p>
 * 支持可选的 EmbeddingModel，当设置了 EmbeddingModel 时：
 * - add/upsert 时会自动对没有 embedding 的文档进行向量化
 * - similaritySearch 支持直接传入文本进行搜索
 */
@Slf4j
public class DefaultMilvusVectorStore implements MilvusVectorStore {

    private final MilvusClientV2 client;
    private final String collectionName;
    private final EmbeddingModel embeddingModel;

    private static final Gson GSON = new Gson();


    public DefaultMilvusVectorStore(MilvusClientV2 client, String collectionName) {
        this(client, collectionName, null);
    }

    public DefaultMilvusVectorStore(MilvusClientV2 client, String collectionName, EmbeddingModel embeddingModel) {
        this.client = client;
        this.collectionName = collectionName;
        this.embeddingModel = embeddingModel;
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
            // 自动嵌入：对没有 embedding 的文档进行向量化
            List<Document> processedDocs = embedDocumentsIfNeeded(documents);

            List<JsonObject> data = processedDocs.stream()
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
            // 自动嵌入：对没有 embedding 的文档进行向量化
            List<Document> processedDocs = embedDocumentsIfNeeded(documents);

            List<JsonObject> data = processedDocs.stream()
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
        return getById(ids, (String) null);
    }

    @Override
    public <T extends Document> List<T> getById(List<String> ids, Class<T> clazz) {
        return getById(ids, null, clazz);
    }

    @Override
    public List<Document> getById(List<String> ids, String partitionName) {
        return getById(ids, partitionName, Document.class);
    }

    @Override
    public <T extends Document> List<T> getById(List<String> ids, String partitionName, Class<T> clazz) {
        if (ids == null || ids.isEmpty()) {
            return Collections.emptyList();
        }
        try {
            GetReq.GetReqBuilder<?, ?> builder = GetReq.builder()
                    .collectionName(collectionName)
                    .ids(new ArrayList<>(ids));

            if (partitionName != null && !partitionName.isEmpty()) {
                builder.partitionName(partitionName);
            }

            GetResp response = client.get(builder.build());
            return response.getGetResults().stream()
                    .map(result -> resultToDocument(result, clazz))
                    .collect(Collectors.toList());
        } catch (Exception e) {
            throw new MilvusDataException(ErrorCode.DATA_QUERY_FAILED,
                    "Failed to get documents by ids from collection: " + collectionName, e);
        }
    }

    @Override
    public List<Document> query(String filterExpression) {
        return query(filterExpression, null, 0, 1000);
    }

    @Override
    public <T extends Document> List<T> query(String filterExpression, Class<T> clazz) {
        return query(filterExpression, null, 0, 1000, clazz);
    }

    @Override
    public List<Document> query(String filterExpression, String partitionName) {
        return query(filterExpression, partitionName, 0, 1000);
    }

    @Override
    public <T extends Document> List<T> query(String filterExpression, String partitionName, Class<T> clazz) {
        return query(filterExpression, partitionName, 0, 1000, clazz);
    }

    @Override
    public List<Document> query(String filterExpression, int offset, int limit) {
        return query(filterExpression, null, offset, limit);
    }

    @Override
    public <T extends Document> List<T> query(String filterExpression, int offset, int limit, Class<T> clazz) {
        return query(filterExpression, null, offset, limit, clazz);
    }

    @Override
    public List<Document> query(String filterExpression, String partitionName, int offset, int limit) {
        return query(filterExpression, partitionName, offset, limit, Document.class);
    }

    @Override
    public <T extends Document> List<T> query(String filterExpression, String partitionName, int offset, int limit, Class<T> clazz) {
        try {
            QueryReq.QueryReqBuilder<?, ?> builder = QueryReq.builder()
                    .collectionName(collectionName)
                    .filter(filterExpression)
                    .offset(offset)
                    .limit(limit);

            if (partitionName != null && !partitionName.isEmpty()) {
                builder.partitionNames(Collections.singletonList(partitionName));
            }

            QueryResp response = client.query(builder.build());
            return response.getQueryResults().stream()
                    .map(result -> queryResultToDocument(result, clazz))
                    .collect(Collectors.toList());
        } catch (Exception e) {
            throw new MilvusDataException(ErrorCode.DATA_QUERY_FAILED,
                    "Failed to query documents from collection: " + collectionName, e);
        }
    }

    // ==================== 向量搜索 - 全局 ====================

    @Override
    @SuppressWarnings({"unchecked", "rawtypes"})
    public List<SearchResult> similaritySearch(SearchRequest request) {
        return (List) doSearch(request, null, Document.class);
    }

    @Override
    public <T extends Document> List<SearchResult<T>> similaritySearch(SearchRequest request, Class<T> clazz) {
        return doSearch(request, null, clazz);
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
    @SuppressWarnings({"unchecked", "rawtypes"})
    public List<SearchResult> similaritySearchInPartition(SearchRequest request, String partitionName) {
        return (List) doSearch(request, Collections.singletonList(partitionName), Document.class);
    }

    @Override
    public <T extends Document> List<SearchResult<T>> similaritySearchInPartition(SearchRequest request, String partitionName, Class<T> clazz) {
        return doSearch(request, Collections.singletonList(partitionName), clazz);
    }

    @Override
    public List<SearchResult> similaritySearchInPartition(List<Float> vector, int topK, String partitionName) {
        return similaritySearchInPartition(SearchRequest.of(vector, topK), partitionName);
    }

    @Override
    @SuppressWarnings({"unchecked", "rawtypes"})
    public List<SearchResult> similaritySearchInPartitions(SearchRequest request, List<String> partitionNames) {
        return (List) doSearch(request, partitionNames, Document.class);
    }

    @Override
    public <T extends Document> List<SearchResult<T>> similaritySearchInPartitions(SearchRequest request, List<String> partitionNames, Class<T> clazz) {
        return doSearch(request, partitionNames, clazz);
    }

    @Override
    public List<SearchResult> similaritySearchInPartitions(List<Float> vector, int topK, List<String> partitionNames) {
        return similaritySearchInPartitions(SearchRequest.of(vector, topK), partitionNames);
    }

    // ==================== 文本搜索（需要 EmbeddingModel）====================

    @Override
    public List<SearchResult> similaritySearch(String query, int topK) {
        return similaritySearch(query, topK, (String) null);
    }

    @Override
    public <T extends Document> List<SearchResult<T>> similaritySearch(String query, int topK, Class<T> clazz) {
        return similaritySearch(query, topK, null, clazz);
    }

    @Override
    @SuppressWarnings({"unchecked", "rawtypes"})
    public List<SearchResult> similaritySearch(String query, int topK, String filter) {
        List<Float> vector = embedQuery(query);
        SearchRequest request = SearchRequest.builder()
                .vector(vector)
                .topK(topK)
                .filter(filter)
                .build();
        return (List) doSearch(request, null, Document.class);
    }

    @Override
    public <T extends Document> List<SearchResult<T>> similaritySearch(String query, int topK, String filter, Class<T> clazz) {
        List<Float> vector = embedQuery(query);
        SearchRequest request = SearchRequest.builder()
                .vector(vector)
                .topK(topK)
                .filter(filter)
                .build();
        return doSearch(request, null, clazz);
    }

    @Override
    @SuppressWarnings({"unchecked", "rawtypes"})
    public List<SearchResult> similaritySearchInPartition(String query, int topK, String partitionName) {
        List<Float> vector = embedQuery(query);
        return (List) doSearch(SearchRequest.of(vector, topK), Collections.singletonList(partitionName), Document.class);
    }

    @Override
    public <T extends Document> List<SearchResult<T>> similaritySearchInPartition(String query, int topK, String partitionName, Class<T> clazz) {
        List<Float> vector = embedQuery(query);
        return doSearch(SearchRequest.of(vector, topK), Collections.singletonList(partitionName), clazz);
    }

    @Override
    @SuppressWarnings({"unchecked", "rawtypes"})
    public List<SearchResult> similaritySearchInPartitions(String query, int topK, List<String> partitionNames) {
        List<Float> vector = embedQuery(query);
        return (List) doSearch(SearchRequest.of(vector, topK), partitionNames, Document.class);
    }

    @Override
    public <T extends Document> List<SearchResult<T>> similaritySearchInPartitions(String query, int topK, List<String> partitionNames, Class<T> clazz) {
        List<Float> vector = embedQuery(query);
        return doSearch(SearchRequest.of(vector, topK), partitionNames, clazz);
    }

    private <T extends Document> List<SearchResult<T>> doSearch(SearchRequest request, List<String> partitionNames, Class<T> clazz) {
        try {
            // 获取需要返回的字段列表（排除 embedding 等带 @ExcludeField 注解的字段）
            List<String> outputFields = Document.getOutputFields(clazz);

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

            List<SearchResult<T>> results = new ArrayList<>();
            for (List<SearchResp.SearchResult> searchResults : response.getSearchResults()) {
                for (SearchResp.SearchResult result : searchResults) {
                    T doc = searchResultToDocument(result, clazz);
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

    /**
     * 对文档进行嵌入处理（如果需要）
     * <p>
     * 如果文档没有 embedding 且设置了 embeddingModel，则自动进行向量化
     */
    private List<Document> embedDocumentsIfNeeded(List<Document> documents) {
        if (embeddingModel == null) {
            return documents;
        }

        // 找出需要嵌入的文档
        List<Document> docsNeedEmbedding = new ArrayList<>();
        List<Integer> indices = new ArrayList<>();

        for (int i = 0; i < documents.size(); i++) {
            Document doc = documents.get(i);
            if (doc.getEmbedding() == null || doc.getEmbedding().isEmpty()) {
                if (doc.getContent() != null && !doc.getContent().isEmpty()) {
                    docsNeedEmbedding.add(doc);
                    indices.add(i);
                }
            }
        }

        if (docsNeedEmbedding.isEmpty()) {
            return documents;
        }

        // 批量获取嵌入向量
        List<String> texts = docsNeedEmbedding.stream()
                .map(Document::getContent)
                .collect(Collectors.toList());

        // EmbeddingModel.embed(List<String>) 返回 List<float[]>
        List<float[]> embeddingsList = embeddingModel.embed(texts);

        // 将嵌入向量设置回文档
        List<Document> result = new ArrayList<>(documents);
        for (int i = 0; i < indices.size(); i++) {
            int docIndex = indices.get(i);
            Document originalDoc = result.get(docIndex);

            // 提取该文档的向量
            float[] embeddingArray = embeddingsList.get(i);
            List<Float> vector = new ArrayList<>(embeddingArray.length);
            for (float f : embeddingArray) {
                vector.add(f);
            }

            // 直接设置 embedding（支持子类）
            originalDoc.setEmbedding(vector);
        }

        log.debug("Embedded {} documents using EmbeddingModel", indices.size());
        return result;
    }

    /**
     * 将查询文本转换为向量
     */
    private List<Float> embedQuery(String query) {
        if (embeddingModel == null) {
            throw new MilvusSearchException(ErrorCode.SEARCH_FAILED,
                    "EmbeddingModel is required for text-based search. Please provide an EmbeddingModel when creating the VectorStore.", null);
        }

        float[] embedding = embeddingModel.embed(query);
        List<Float> vector = new ArrayList<>(embedding.length);
        for (float f : embedding) {
            vector.add(f);
        }
        return vector;
    }

    private JsonObject documentToJsonObject(Document document) {
        return document.toJsonObject();
    }

    private <T extends Document> T resultToDocument(QueryResp.QueryResult result, Class<T> clazz) {
        Map<String, Object> entity = result.getEntity();
        return entityToDocument(entity, clazz);
    }

    private <T extends Document> T queryResultToDocument(QueryResp.QueryResult result, Class<T> clazz) {
        Map<String, Object> entity = result.getEntity();
        return entityToDocument(entity, clazz);
    }

    private <T extends Document> T searchResultToDocument(SearchResp.SearchResult result, Class<T> clazz) {
        Map<String, Object> entity = result.getEntity();
        return entityToDocument(entity, clazz);
    }


    private <T extends Document> T entityToDocument(Map<String, Object> entity, Class<T> clazz) {
        // 使用 Gson 直接将 entity 反序列化为指定类型
        String json = GSON.toJson(entity);
        return GSON.fromJson(json, clazz);
    }
}

