package com.example.demo;

import com.fyj.rag.client.MilvusClient;
import com.fyj.rag.exception.MilvusCollectionException;
import com.fyj.rag.exception.MilvusPartitionException;
import com.fyj.rag.schema.CollectionSchema;
import com.fyj.rag.schema.FieldSchema;
import com.fyj.rag.schema.IndexSchema;
import com.fyj.rag.vectorstore.Document;
import com.fyj.rag.vectorstore.MilvusVectorStore;
import com.fyj.rag.vectorstore.SearchRequest;
import com.fyj.rag.vectorstore.SearchResult;
import io.milvus.v2.common.IndexParam;
import org.junit.jupiter.api.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Milvus Vector Store 完整测试用例
 *
 * 注意：运行测试前需要启动 Milvus 服务
 * docker run -d --name milvus -p 19530:19530 milvusdb/milvus:latest
 */
@SpringBootTest
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class MilvusVectorStoreTests {

    @Autowired
    private MilvusClient milvusClient;

    private static final String TEST_COLLECTION = "test_vector_store";
    private static final String TEST_PARTITION_1 = "partition_tech";
    private static final String TEST_PARTITION_2 = "partition_news";
    private static final int DIMENSION = 128;

    // ==================== Collection 管理测试 ====================

    @Test
    @Order(1)
    @DisplayName("测试创建 Collection（使用自定义 Schema）")
    void testCreateCollectionWithCustomSchema() {
        // 如果已存在，先删除
        if (milvusClient.hasCollection(TEST_COLLECTION)) {
            milvusClient.dropCollection(TEST_COLLECTION);
        }

        // 创建自定义 Schema
        CollectionSchema schema = CollectionSchema.create()
                .description("Test collection for vector store")
                .primaryKey("id")
                .varchar("content", 65535)
                .floatVector("embedding", DIMENSION)
                .json("metadata")
                .enableDynamicField(true)
                .build();

        // 创建索引
        IndexSchema indexSchema = IndexSchema.autoIndex("embedding", IndexParam.MetricType.COSINE);

        // 创建 Collection
        milvusClient.createCollection(TEST_COLLECTION, schema, indexSchema);

        // 验证
        assertTrue(milvusClient.hasCollection(TEST_COLLECTION));
        System.out.println("✅ Collection created successfully: " + TEST_COLLECTION);
    }

    @Test
    @Order(2)
    @DisplayName("测试加载 Collection")
    void testLoadCollection() {
        milvusClient.loadCollection(TEST_COLLECTION);

        Boolean loadState = milvusClient.getLoadState(TEST_COLLECTION);
        assertTrue(loadState);
        System.out.println("✅ Collection loaded successfully");
    }

    @Test
    @Order(3)
    @DisplayName("测试列出所有 Collection")
    void testListCollections() {
        List<String> collections = milvusClient.listCollections();

        assertNotNull(collections);
        assertTrue(collections.contains(TEST_COLLECTION));
        System.out.println("✅ Collections: " + collections);
    }

    @Test
    @Order(4)
    @DisplayName("测试获取 Collection 详情")
    void testDescribeCollection() {
        var description = milvusClient.describeCollection(TEST_COLLECTION);

        assertNotNull(description);
        assertEquals(TEST_COLLECTION, description.getCollectionName());
        System.out.println("✅ Collection description: " + description.getCollectionName());
        System.out.println("   Fields: " + description.getFieldNames());
    }

    // ==================== 分区管理测试 ====================

    @Test
    @Order(10)
    @DisplayName("测试创建分区")
    void testCreatePartition() {
        MilvusVectorStore vectorStore = milvusClient.getVectorStore(TEST_COLLECTION);

        // 创建分区
        vectorStore.createPartition(TEST_PARTITION_1);
        vectorStore.createPartition(TEST_PARTITION_2);

        // 验证
        assertTrue(vectorStore.hasPartition(TEST_PARTITION_1));
        assertTrue(vectorStore.hasPartition(TEST_PARTITION_2));
        System.out.println("✅ Partitions created: " + TEST_PARTITION_1 + ", " + TEST_PARTITION_2);
    }

    @Test
    @Order(11)
    @DisplayName("测试列出所有分区")
    void testListPartitions() {
        MilvusVectorStore vectorStore = milvusClient.getVectorStore(TEST_COLLECTION);

        List<String> partitions = vectorStore.listPartitions();

        assertNotNull(partitions);
        assertTrue(partitions.contains(TEST_PARTITION_1));
        assertTrue(partitions.contains(TEST_PARTITION_2));
        assertTrue(partitions.contains("_default")); // 默认分区
        System.out.println("✅ Partitions: " + partitions);
    }

    @Test
    @Order(12)
    @DisplayName("测试加载分区")
    void testLoadPartition() {
        MilvusVectorStore vectorStore = milvusClient.getVectorStore(TEST_COLLECTION);

        // 加载分区
        vectorStore.loadPartitions(Arrays.asList(TEST_PARTITION_1, TEST_PARTITION_2));
        System.out.println("✅ Partitions loaded");
    }

    // ==================== 数据操作测试 ====================

    @Test
    @Order(20)
    @DisplayName("测试添加文档到默认分区")
    void testAddDocumentsToDefaultPartition() {
        MilvusVectorStore vectorStore = milvusClient.getVectorStore(TEST_COLLECTION);

        List<Document> documents = createTestDocuments("default", 5);
        vectorStore.add(documents);

        // 等待数据刷新
        vectorStore.flush();

        System.out.println("✅ Added 5 documents to default partition");
    }

    @Test
    @Order(21)
    @DisplayName("测试添加文档到指定分区")
    void testAddDocumentsToPartition() {
        MilvusVectorStore vectorStore = milvusClient.getVectorStore(TEST_COLLECTION);

        // 添加文档到 partition_tech
        List<Document> techDocs = createTestDocuments("tech", 3);
        vectorStore.add(techDocs, TEST_PARTITION_1);

        // 添加文档到 partition_news
        List<Document> newsDocs = createTestDocuments("news", 3);
        vectorStore.add(newsDocs, TEST_PARTITION_2);

        vectorStore.flush();

        System.out.println("✅ Added 3 documents to " + TEST_PARTITION_1);
        System.out.println("✅ Added 3 documents to " + TEST_PARTITION_2);
    }

    @Test
    @Order(22)
    @DisplayName("测试统计文档数量")
    void testCountDocuments() {
        MilvusVectorStore vectorStore = milvusClient.getVectorStore(TEST_COLLECTION);

        long totalCount = vectorStore.count();
        long partition1Count = vectorStore.count(TEST_PARTITION_1);
        long partition2Count = vectorStore.count(TEST_PARTITION_2);

        System.out.println("✅ Total documents: " + totalCount);
        System.out.println("   Partition " + TEST_PARTITION_1 + ": " + partition1Count);
        System.out.println("   Partition " + TEST_PARTITION_2 + ": " + partition2Count);

        assertTrue(totalCount >= 11); // 5 + 3 + 3
    }

    @Test
    @Order(23)
    @DisplayName("测试根据 ID 获取文档")
    void testGetById() {
        MilvusVectorStore vectorStore = milvusClient.getVectorStore(TEST_COLLECTION);

        List<Document> documents = vectorStore.getById(Arrays.asList("default_0", "default_1"));

        assertNotNull(documents);
        assertFalse(documents.isEmpty());
        System.out.println("✅ Retrieved documents by ID: " + documents.size());
        documents.forEach(doc -> System.out.println("   - " + doc.getId() + ": " + doc.getContent()));
    }

    @Test
    @Order(24)
    @DisplayName("测试根据过滤条件查询")
    void testQueryByFilter() {
        MilvusVectorStore vectorStore = milvusClient.getVectorStore(TEST_COLLECTION);

        // 查询 content 包含 "tech" 的文档
        List<Document> documents = vectorStore.query("id like \"tech%\"", 10);

        assertNotNull(documents);
        System.out.println("✅ Query results: " + documents.size());
        documents.forEach(doc -> System.out.println("   - " + doc.getId() + ": " + doc.getContent()));
    }

    // ==================== 向量搜索测试 ====================

    @Test
    @Order(30)
    @DisplayName("测试向量相似度搜索（全局）")
    void testSimilaritySearch() {
        MilvusVectorStore vectorStore = milvusClient.getVectorStore(TEST_COLLECTION);

        // 创建查询向量
        List<Float> queryVector = createRandomVector(DIMENSION);

        // 搜索
        List<SearchResult> results = vectorStore.similaritySearch(queryVector, 5);

        assertNotNull(results);
        assertFalse(results.isEmpty());
        System.out.println("✅ Similarity search results: " + results.size());
        results.forEach(r -> System.out.println("   - " + r.getDocument().getId()
                + " (score: " + r.getScore() + ")"));
    }

    @Test
    @Order(31)
    @DisplayName("测试向量相似度搜索（带过滤条件）")
    void testSimilaritySearchWithFilter() {
        MilvusVectorStore vectorStore = milvusClient.getVectorStore(TEST_COLLECTION);

        List<Float> queryVector = createRandomVector(DIMENSION);

        // 只搜索 id 以 "tech" 开头的文档
        List<SearchResult> results = vectorStore.similaritySearch(queryVector, 5, "id like \"tech%\"");

        assertNotNull(results);
        System.out.println("✅ Filtered search results: " + results.size());
        results.forEach(r -> {
            assertTrue(r.getDocument().getId().startsWith("tech"));
            System.out.println("   - " + r.getDocument().getId() + " (score: " + r.getScore() + ")");
        });
    }

    @Test
    @Order(32)
    @DisplayName("测试在指定分区搜索")
    void testSimilaritySearchInPartition() {
        MilvusVectorStore vectorStore = milvusClient.getVectorStore(TEST_COLLECTION);

        List<Float> queryVector = createRandomVector(DIMENSION);

        // 只在 partition_tech 中搜索
        List<SearchResult> results = vectorStore.similaritySearchInPartition(queryVector, 5, TEST_PARTITION_1);

        assertNotNull(results);
        System.out.println("✅ Search in partition " + TEST_PARTITION_1 + ": " + results.size());
        results.forEach(r -> System.out.println("   - " + r.getDocument().getId()
                + " (score: " + r.getScore() + ")"));
    }

    @Test
    @Order(33)
    @DisplayName("测试在多个分区搜索")
    void testSimilaritySearchInMultiplePartitions() {
        MilvusVectorStore vectorStore = milvusClient.getVectorStore(TEST_COLLECTION);

        List<Float> queryVector = createRandomVector(DIMENSION);

        // 在多个分区中搜索
        List<SearchResult> results = vectorStore.similaritySearchInPartitions(
                queryVector, 10, Arrays.asList(TEST_PARTITION_1, TEST_PARTITION_2));

        assertNotNull(results);
        System.out.println("✅ Search in multiple partitions: " + results.size());
        results.forEach(r -> System.out.println("   - " + r.getDocument().getId()
                + " (score: " + r.getScore() + ")"));
    }

    @Test
    @Order(34)
    @DisplayName("测试使用 SearchRequest 搜索")
    void testSimilaritySearchWithRequest() {
        MilvusVectorStore vectorStore = milvusClient.getVectorStore(TEST_COLLECTION);

        List<Float> queryVector = createRandomVector(DIMENSION);

        SearchRequest request = SearchRequest.builder()
                .vector(queryVector)
                .topK(5)
                .similarityThreshold(0.0f)
                .outputFields(Arrays.asList("id", "content", "metadata"))
                .build();

        List<SearchResult> results = vectorStore.similaritySearch(request);

        assertNotNull(results);
        System.out.println("✅ Search with request: " + results.size());
        results.forEach(r -> System.out.println("   - " + r.getDocument().getId()
                + " (score: " + r.getScore() + ")"));
    }

    // ==================== Upsert 测试 ====================

    @Test
    @Order(40)
    @DisplayName("测试 Upsert 操作")
    void testUpsert() {
        MilvusVectorStore vectorStore = milvusClient.getVectorStore(TEST_COLLECTION);

        // 创建一个已存在的文档（更新）和一个新文档
        List<Document> documents = Arrays.asList(
                Document.builder()
                        .id("default_0")  // 已存在
                        .content("Updated content for default_0")
                        .embedding(createRandomVector(DIMENSION))
                        .metadata(Map.of("updated", true))
                        .build(),
                Document.builder()
                        .id("upsert_new_1")  // 新文档
                        .content("New document via upsert")
                        .embedding(createRandomVector(DIMENSION))
                        .metadata(Map.of("source", "upsert"))
                        .build()
        );

        vectorStore.upsert(documents);
        vectorStore.flush();

        // 验证
        List<Document> retrieved = vectorStore.getById(Arrays.asList("default_0", "upsert_new_1"));
        assertFalse(retrieved.isEmpty());
        System.out.println("✅ Upsert completed");
    }

    // ==================== 删除测试 ====================

    @Test
    @Order(50)
    @DisplayName("测试根据 ID 删除文档")
    void testDeleteById() {
        MilvusVectorStore vectorStore = milvusClient.getVectorStore(TEST_COLLECTION);

        // 先添加一个测试文档
        Document doc = Document.builder()
                .id("to_delete_1")
                .content("This document will be deleted")
                .embedding(createRandomVector(DIMENSION))
                .build();
        vectorStore.add(Collections.singletonList(doc));
        vectorStore.flush();

        // 删除
        vectorStore.delete(Collections.singletonList("to_delete_1"));
        vectorStore.flush();

        // 验证
        List<Document> result = vectorStore.getById(Collections.singletonList("to_delete_1"));
        assertTrue(result.isEmpty());
        System.out.println("✅ Document deleted by ID");
    }

    @Test
    @Order(51)
    @DisplayName("测试根据过滤条件删除文档")
    void testDeleteByFilter() {
        MilvusVectorStore vectorStore = milvusClient.getVectorStore(TEST_COLLECTION);

        // 先添加测试文档
        List<Document> docs = Arrays.asList(
                Document.builder()
                        .id("filter_delete_1")
                        .content("Delete by filter 1")
                        .embedding(createRandomVector(DIMENSION))
                        .build(),
                Document.builder()
                        .id("filter_delete_2")
                        .content("Delete by filter 2")
                        .embedding(createRandomVector(DIMENSION))
                        .build()
        );
        vectorStore.add(docs);
        vectorStore.flush();

        // 删除
        vectorStore.deleteByFilter("id like \"filter_delete%\"");
        vectorStore.flush();

        System.out.println("✅ Documents deleted by filter");
    }

    // ==================== 异常测试 ====================

    @Test
    @Order(60)
    @DisplayName("测试操作不存在的 Collection 抛出异常")
    void testCollectionNotFoundThrowsException() {
        assertThrows(MilvusCollectionException.class, () -> {
            milvusClient.describeCollection("non_existent_collection");
        });
        System.out.println("✅ MilvusCollectionException thrown as expected");
    }

    @Test
    @Order(61)
    @DisplayName("测试创建已存在的分区抛出异常")
    void testCreateExistingPartitionThrowsException() {
        MilvusVectorStore vectorStore = milvusClient.getVectorStore(TEST_COLLECTION);

        // 分区已存在，应该抛出异常
        assertThrows(MilvusPartitionException.class, () -> {
            vectorStore.createPartition(TEST_PARTITION_1);
        });
        System.out.println("✅ MilvusPartitionException thrown as expected");
    }

    // ==================== 清理测试 ====================

    @Test
    @Order(100)
    @DisplayName("测试删除分区")
    void testDropPartition() {
        MilvusVectorStore vectorStore = milvusClient.getVectorStore(TEST_COLLECTION);

        // 先释放分区
        vectorStore.releasePartition(TEST_PARTITION_1);
        vectorStore.releasePartition(TEST_PARTITION_2);

        // 删除分区
        vectorStore.dropPartition(TEST_PARTITION_1);
        vectorStore.dropPartition(TEST_PARTITION_2);

        assertFalse(vectorStore.hasPartition(TEST_PARTITION_1));
        assertFalse(vectorStore.hasPartition(TEST_PARTITION_2));
        System.out.println("✅ Partitions dropped");
    }

    @Test
    @Order(101)
    @DisplayName("测试删除 Collection")
    void testDropCollection() {
        milvusClient.releaseCollection(TEST_COLLECTION);
        milvusClient.dropCollection(TEST_COLLECTION);

        assertFalse(milvusClient.hasCollection(TEST_COLLECTION));
        System.out.println("✅ Collection dropped: " + TEST_COLLECTION);
    }

    // ==================== 辅助方法 ====================

    /**
     * 创建测试文档
     */
    private List<Document> createTestDocuments(String prefix, int count) {
        return IntStream.range(0, count)
                .mapToObj(i -> Document.builder()
                        .id(prefix + "_" + i)
                        .content("This is test document " + prefix + "_" + i)
                        .embedding(createRandomVector(DIMENSION))
                        .metadata(Map.of(
                                "index", i,
                                "category", prefix,
                                "timestamp", System.currentTimeMillis()
                        ))
                        .build())
                .collect(Collectors.toList());
    }

    /**
     * 创建随机向量
     */
    private List<Float> createRandomVector(int dimension) {
        Random random = new Random();
        List<Float> vector = new ArrayList<>(dimension);
        for (int i = 0; i < dimension; i++) {
            vector.add(random.nextFloat());
        }
        // 归一化
        float norm = 0;
        for (Float f : vector) {
            norm += f * f;
        }
        norm = (float) Math.sqrt(norm);
        for (int i = 0; i < vector.size(); i++) {
            vector.set(i, vector.get(i) / norm);
        }
        return vector;
    }
}

