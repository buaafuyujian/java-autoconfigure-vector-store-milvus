package com.example.demo;

import com.example.demo.entity.DocumentSegment;
import com.fyj.rag.client.MilvusClient;
import com.fyj.rag.vectorstore.Document;
import com.fyj.rag.vectorstore.MilvusVectorStore;
import com.fyj.rag.vectorstore.SearchResult;
import org.junit.jupiter.api.*;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * DocumentSegment 测试用例 - 使用 EmbeddingModel 自动嵌入
 * <p>
 * 特点：
 * 1. add/upsert 时不需要手动设置 embedding，VectorStore 会自动调用 EmbeddingModel
 * 2. 搜索时可以直接传入文本，VectorStore 会自动转换为向量
 */
@SpringBootTest
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class DocumentSegmentTests {

    @Autowired
    private MilvusClient milvusClient;

    @Autowired
    private EmbeddingModel embeddingModel;

    private static MilvusVectorStore vectorStore;

    private static final String KNOWLEDGE_1 = "kb001";
    private static final String KNOWLEDGE_2 = "kb002";
    private static final String FILE_1 = "file_001";
    private static final String FILE_2 = "file_002";
    private static final String FILE_3 = "file_003";

    @BeforeAll
    static void setup(@Autowired MilvusClient client, @Autowired EmbeddingModel embeddingModel) {
        // 使用带 EmbeddingModel 的 VectorStore，支持自动嵌入
        vectorStore = client.getVectorStore(DocumentSegment.COLLECTION_NAME, embeddingModel);
    }

    // ==================== 1. Collection 和分区初始化 ====================

    @Test
    @Order(1)
    @DisplayName("1.1 初始化 Collection")
    void testInitCollection() {
        if (milvusClient.hasCollection(DocumentSegment.COLLECTION_NAME)) {
            milvusClient.releaseCollection(DocumentSegment.COLLECTION_NAME);
            milvusClient.dropCollection(DocumentSegment.COLLECTION_NAME);
        }

        // 使用 EmbeddingModel 的维度创建 Schema
        int dimension = embeddingModel.dimensions();
        System.out.println("   EmbeddingModel 维度: " + dimension);

        milvusClient.createCollection(
                DocumentSegment.COLLECTION_NAME,
                DocumentSegment.createSchema(dimension),
                DocumentSegment.createIndex()
        );
        milvusClient.loadCollection(DocumentSegment.COLLECTION_NAME);

        assertTrue(milvusClient.hasCollection(DocumentSegment.COLLECTION_NAME));
        System.out.println("✅ Collection 创建成功: " + DocumentSegment.COLLECTION_NAME);
    }

    @Test
    @Order(2)
    @DisplayName("1.2 为每个知识库创建分区")
    void testCreatePartitions() {
        String partition1 = DocumentSegment.getPartitionName(KNOWLEDGE_1);
        vectorStore.createPartition(partition1);
        System.out.println("✅ 创建分区: " + partition1);

        String partition2 = DocumentSegment.getPartitionName(KNOWLEDGE_2);
        vectorStore.createPartition(partition2);
        System.out.println("✅ 创建分区: " + partition2);

        System.out.println("   所有分区: " + vectorStore.listPartitions());
    }

    // ==================== 2. 按分区插入数据（自动嵌入）====================

    @Test
    @Order(10)
    @DisplayName("2.1 向知识库1插入数据（自动嵌入，无需手动设置 embedding）")
    void testInsertToPartition1() {
        String partition = DocumentSegment.getPartitionName(KNOWLEDGE_1);

        // 创建文档时不需要设置 embedding，VectorStore 会自动调用 EmbeddingModel
        List<DocumentSegment> segments1 = createTestSegments(FILE_1, 0, 3);
        vectorStore.add(new ArrayList<>(segments1), partition);
        System.out.println("✅ 插入 " + segments1.size() + " 个片段到 " + partition + " (文档: " + FILE_1 + ")");

        List<DocumentSegment> segments2 = createTestSegments(FILE_2, 0, 2);
        vectorStore.add(new ArrayList<>(segments2), partition);
        System.out.println("✅ 插入 " + segments2.size() + " 个片段到 " + partition + " (文档: " + FILE_2 + ")");
    }

    @Test
    @Order(11)
    @DisplayName("2.2 向知识库2插入数据")
    void testInsertToPartition2() {
        String partition = DocumentSegment.getPartitionName(KNOWLEDGE_2);

        List<DocumentSegment> segments = createTestSegments(FILE_3, 0, 3);
        vectorStore.add(new ArrayList<>(segments), partition);
        System.out.println("✅ 插入 " + segments.size() + " 个片段到 " + partition + " (文档: " + FILE_3 + ")");
    }

    @Test
    @Order(12)
    @DisplayName("2.3 统计各分区数据量")
    void testCountByPartition() {
        String partition1 = DocumentSegment.getPartitionName(KNOWLEDGE_1);
        String partition2 = DocumentSegment.getPartitionName(KNOWLEDGE_2);

        long count1 = vectorStore.count(partition1);
        long count2 = vectorStore.count(partition2);
        long totalCount = vectorStore.count();

        System.out.println("✅ 数据统计:");
        System.out.println("   分区 " + partition1 + ": " + count1 + " 条");
        System.out.println("   分区 " + partition2 + ": " + count2 + " 条");
        System.out.println("   总计: " + totalCount + " 条");

        assertEquals(5, count1);  // 3 + 2
        assertEquals(3, count2);
    }

    // ==================== 3. 查询操作 ====================

    @Test
    @Order(20)
    @DisplayName("3.1 在知识库分区中按文档ID查询")
    void testQueryInPartition() {
        String partition = DocumentSegment.getPartitionName(KNOWLEDGE_1);
        String filter = DocumentSegment.filterByFileId(FILE_1);

        List<Document> results = vectorStore.query(filter, partition, 0, 100);

        assertFalse(results.isEmpty());

        List<DocumentSegment> segments = results.stream()
                .map(DocumentSegment::fromDocument)
                .toList();

        segments.forEach(s -> assertEquals(FILE_1, s.getFileId()));
        System.out.println("✅ 查询文档 " + FILE_1 + "，返回 " + segments.size() + " 条");
        segments.forEach(s -> System.out.println("   - " + s.getId() + ": " + s.getContent()));
    }

    @Test
    @Order(21)
    @DisplayName("3.2 根据ID获取数据")
    void testGetByIdInPartition() {
        String partition = DocumentSegment.getPartitionName(KNOWLEDGE_1);
        String id = FILE_1 + "_0";

        List<Document> results = vectorStore.getById(Collections.singletonList(id), partition);

        assertFalse(results.isEmpty());
        DocumentSegment segment = DocumentSegment.fromDocument(results.get(0));

        assertEquals(id, segment.getId());
        assertEquals(FILE_1, segment.getFileId());

        System.out.println("✅ 获取成功: " + segment.getId());
        System.out.println("   fileId: " + segment.getFileId());
        System.out.println("   content: " + segment.getContent());
    }

    // ==================== 4. 文本搜索（自动嵌入查询）====================

    @Test
    @Order(30)
    @DisplayName("4.1 使用文本搜索（自动转换为向量）")
    void testTextSearch() {
        String partition = DocumentSegment.getPartitionName(KNOWLEDGE_1);

        // 直接使用文本搜索，VectorStore 会自动调用 EmbeddingModel 转换为向量
        String query = "Java 编程语言";
        List<SearchResult> results = vectorStore.similaritySearchInPartition(query, 3, partition);

        assertFalse(results.isEmpty());

        System.out.println("✅ 文本搜索 \"" + query + "\"，返回 " + results.size() + " 条:");
        results.forEach(r -> {
            DocumentSegment seg = DocumentSegment.fromDocument(r.getDocument());
            System.out.println("   - " + seg.getId() + " (score: " + String.format("%.4f", r.getScore()) + ")");
            System.out.println("     content: " + seg.getContent());
        });
    }

    @Test
    @Order(31)
    @DisplayName("4.2 在多个知识库中文本搜索")
    void testTextSearchInMultiplePartitions() {
        List<String> partitions = DocumentSegment.getPartitionNames(Arrays.asList(KNOWLEDGE_1, KNOWLEDGE_2));

        String query = "Spring Boot 框架";
        List<SearchResult> results = vectorStore.similaritySearchInPartitions(query, 5, partitions);

        assertFalse(results.isEmpty());

        System.out.println("✅ 跨知识库搜索 \"" + query + "\"，返回 " + results.size() + " 条:");
        results.forEach(r -> {
            DocumentSegment seg = DocumentSegment.fromDocument(r.getDocument());
            System.out.println("   - " + seg.getId() + " [" + seg.getFileId() + "] (score: " + String.format("%.4f", r.getScore()) + ")");
        });
    }

    @Test
    @Order(32)
    @DisplayName("4.3 全局文本搜索")
    void testGlobalTextSearch() {
        String query = "人工智能技术";
        List<SearchResult> results = vectorStore.similaritySearch(query, 5);

        assertFalse(results.isEmpty());

        System.out.println("✅ 全局搜索 \"" + query + "\"，返回 " + results.size() + " 条:");
        results.forEach(r -> {
            DocumentSegment seg = DocumentSegment.fromDocument(r.getDocument());
            System.out.println("   - " + seg.getId() + " (score: " + String.format("%.4f", r.getScore()) + ")");
        });
    }

    // ==================== 5. 分区数据管理 ====================

    @Test
    @Order(40)
    @DisplayName("5.1 Upsert（自动嵌入）")
    void testUpsertInPartition() {
        String partition = DocumentSegment.getPartitionName(KNOWLEDGE_1);

        // 不需要设置 embedding
        DocumentSegment segment = DocumentSegment.builder()
                .id("upsert_test")
                .fileId(FILE_1)
                .content("这是通过 Upsert 插入的新内容，会自动进行向量化")
                .build();

        vectorStore.upsert(Collections.singletonList(segment), partition);

        List<Document> results = vectorStore.getById(Collections.singletonList("upsert_test"), partition);
        assertFalse(results.isEmpty());

        System.out.println("✅ Upsert 成功（自动嵌入）");
    }

    @Test
    @Order(41)
    @DisplayName("5.2 删除数据")
    void testDeleteInPartition() {
        String partition = DocumentSegment.getPartitionName(KNOWLEDGE_1);

        vectorStore.delete(Collections.singletonList("upsert_test"), partition);

        List<Document> results = vectorStore.getById(Collections.singletonList("upsert_test"), partition);
        assertTrue(results.isEmpty());

        System.out.println("✅ 删除成功");
    }

    @Test
    @Order(42)
    @DisplayName("5.3 根据文档ID删除所有片段")
    void testDeleteByFileId() {
        String partition = DocumentSegment.getPartitionName(KNOWLEDGE_1);

        // 先插入测试数据
        String testFileId = "file_to_delete";
        List<DocumentSegment> segments = createTestSegments(testFileId, 0, 2);
        vectorStore.add(new ArrayList<>(segments), partition);

        long countBefore = vectorStore.count(partition);
        System.out.println("   删除前: " + countBefore);

        // 删除
        String filter = DocumentSegment.filterByFileId(testFileId);
        vectorStore.deleteByFilter(filter, partition);

        long countAfter = vectorStore.count(partition);
        System.out.println("   删除后: " + countAfter);

        assertEquals(countBefore - 2, countAfter);
        System.out.println("✅ 删除文档 " + testFileId + " 成功");
    }

    // ==================== 6. 分区管理 ====================

    @Test
    @Order(50)
    @DisplayName("6.1 创建新知识库分区")
    void testCreateNewPartition() {
        String newKnowledgeId = "kb003";
        String partition = DocumentSegment.getPartitionName(newKnowledgeId);

        vectorStore.createPartition(partition);
        assertTrue(vectorStore.hasPartition(partition));

        // 插入数据（自动嵌入）
        List<DocumentSegment> segments = createTestSegments("new_file", 0, 2);
        vectorStore.add(new ArrayList<>(segments), partition);

        System.out.println("✅ 创建分区: " + partition);
        System.out.println("   所有分区: " + vectorStore.listPartitions());
    }

    @Test
    @Order(51)
    @DisplayName("6.2 删除知识库分区")
    void testDeletePartition() {
        String knowledgeId = "kb003";
        String partition = DocumentSegment.getPartitionName(knowledgeId);

        vectorStore.releasePartition(partition);
        vectorStore.dropPartition(partition);

        assertFalse(vectorStore.hasPartition(partition));
        System.out.println("✅ 删除分区: " + partition);
    }

    // ==================== 7. 清理 ====================

    @Test
    @Order(100)
    @DisplayName("7.1 清理 Collection")
    void testCleanup() {
        milvusClient.releaseCollection(DocumentSegment.COLLECTION_NAME);
        milvusClient.dropCollection(DocumentSegment.COLLECTION_NAME);

        assertFalse(milvusClient.hasCollection(DocumentSegment.COLLECTION_NAME));
        System.out.println("✅ Collection 已删除");
    }

    // ==================== 辅助方法 ====================

    /**
     * 创建测试片段（不设置 embedding，由 VectorStore 自动嵌入）
     */
    private List<DocumentSegment> createTestSegments(String fileId, int startIndex, int count) {
        List<String> contents = Arrays.asList(
                "Java 是一种广泛使用的编程语言，具有跨平台特性",
                "Spring Boot 是一个用于快速构建 Spring 应用的框架",
                "人工智能正在改变我们的生活方式",
                "机器学习是人工智能的一个重要分支",
                "深度学习在图像识别领域取得了巨大成功"
        );

        return IntStream.range(startIndex, startIndex + count)
                .mapToObj(i -> DocumentSegment.builder()
                        .id(fileId + "_" + i)
                        .fileId(fileId)
                        .content(contents.get(i % contents.size()) + " - 片段 " + i)
                        // 不设置 embedding，由 VectorStore 自动调用 EmbeddingModel
                        .metadata(Map.of("chunk_index", i, "total_chunks", count))
                        .build())
                .collect(Collectors.toList());
    }
}

