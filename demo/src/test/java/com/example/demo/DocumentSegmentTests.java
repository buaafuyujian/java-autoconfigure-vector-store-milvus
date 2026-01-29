package com.example.demo;

import com.example.demo.entity.DocumentSegment;
import com.fyj.rag.client.MilvusClient;
import com.fyj.rag.vectorstore.Document;
import com.fyj.rag.vectorstore.MilvusVectorStore;
import com.fyj.rag.vectorstore.SearchResult;
import org.junit.jupiter.api.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * DocumentSegment 测试用例 - 使用分区功能
 * <p>
 * 每个知识库对应一个分区，通过分区来隔离不同知识库的数据
 * DocumentSegment 不存储 knowledgeId，知识库信息通过分区名推断
 */
@SpringBootTest
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class DocumentSegmentTests {

    @Autowired
    private MilvusClient milvusClient;

    private static MilvusVectorStore vectorStore;

    private static final int DIMENSION = 128;
    private static final String KNOWLEDGE_1 = "kb001";  // 知识库1
    private static final String KNOWLEDGE_2 = "kb002";  // 知识库2
    private static final String FILE_1 = "file_001";
    private static final String FILE_2 = "file_002";
    private static final String FILE_3 = "file_003";

    @BeforeAll
    static void setup(@Autowired MilvusClient client) {
        vectorStore = client.getVectorStore(
                DocumentSegment.COLLECTION_NAME,
                DocumentSegment.FIELD_ID,
                DocumentSegment.FIELD_CONTENT,
                DocumentSegment.FIELD_EMBEDDING,
                DocumentSegment.FIELD_METADATA,
                Collections.singletonList(DocumentSegment.FIELD_FILE_ID)
        );
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

        milvusClient.createCollection(
                DocumentSegment.COLLECTION_NAME,
                DocumentSegment.createSchema(DIMENSION),
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
        System.out.println("✅ 创建分区: " + partition1 + " (知识库: " + KNOWLEDGE_1 + ")");

        String partition2 = DocumentSegment.getPartitionName(KNOWLEDGE_2);
        vectorStore.createPartition(partition2);
        System.out.println("✅ 创建分区: " + partition2 + " (知识库: " + KNOWLEDGE_2 + ")");

        List<String> partitions = vectorStore.listPartitions();
        System.out.println("   所有分区: " + partitions);
    }

    // ==================== 2. 按分区插入数据 ====================

    @Test
    @Order(10)
    @DisplayName("2.1 向知识库1的分区插入数据")
    void testInsertToPartition1() {
        String partition = DocumentSegment.getPartitionName(KNOWLEDGE_1);

        List<DocumentSegment> segments1 = createTestSegments(FILE_1, 0, 5);
        vectorStore.add(new ArrayList<>(segments1), partition);
        System.out.println("✅ 向分区 " + partition + " 插入 " + segments1.size() + " 个片段 (文档: " + FILE_1 + ")");

        List<DocumentSegment> segments2 = createTestSegments(FILE_2, 0, 3);
        vectorStore.add(new ArrayList<>(segments2), partition);
        System.out.println("✅ 向分区 " + partition + " 插入 " + segments2.size() + " 个片段 (文档: " + FILE_2 + ")");
    }

    @Test
    @Order(11)
    @DisplayName("2.2 向知识库2的分区插入数据")
    void testInsertToPartition2() {
        String partition = DocumentSegment.getPartitionName(KNOWLEDGE_2);

        List<DocumentSegment> segments = createTestSegments(FILE_3, 0, 4);
        vectorStore.add(new ArrayList<>(segments), partition);
        System.out.println("✅ 向分区 " + partition + " 插入 " + segments.size() + " 个片段 (文档: " + FILE_3 + ")");
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

        assertEquals(8, count1);  // 5 + 3
        assertEquals(4, count2);
    }

    // ==================== 3. 在指定分区中查询 ====================

    @Test
    @Order(20)
    @DisplayName("3.1 在知识库分区中按文档ID查询")
    void testQueryInPartition() {
        String partition = DocumentSegment.getPartitionName(KNOWLEDGE_1);
        String filter = DocumentSegment.filterByFileId(FILE_1);

        List<Document> results = vectorStore.query(filter, partition, 100);

        assertFalse(results.isEmpty());

        List<DocumentSegment> segments = results.stream()
                .map(DocumentSegment::fromDocument)
                .toList();

        segments.forEach(s -> assertEquals(FILE_1, s.getFileId()));
        System.out.println("✅ 在分区 " + partition + " 中查询文档 " + FILE_1 + "，返回 " + segments.size() + " 条");
    }

    @Test
    @Order(21)
    @DisplayName("3.2 根据ID从指定分区获取数据")
    void testGetByIdInPartition() {
        String partition = DocumentSegment.getPartitionName(KNOWLEDGE_1);
        String id = FILE_1 + "_0";

        List<Document> results = vectorStore.getById(Collections.singletonList(id), partition);

        assertFalse(results.isEmpty());
        DocumentSegment segment = DocumentSegment.fromDocument(results.get(0));

        assertEquals(id, segment.getId());
        assertEquals(FILE_1, segment.getFileId());

        System.out.println("✅ 从分区获取: " + segment.getId());
        System.out.println("   fileId: " + segment.getFileId());
        System.out.println("   content: " + segment.getContent());
    }

    // ==================== 4. 在指定分区中向量搜索 ====================

    @Test
    @Order(30)
    @DisplayName("4.1 在单个知识库（分区）中搜索")
    void testSearchInSinglePartition() {
        List<Float> queryVector = createRandomVector(DIMENSION);
        String partition = DocumentSegment.getPartitionName(KNOWLEDGE_1);

        List<SearchResult> results = vectorStore.similaritySearchInPartition(queryVector, 5, partition);

        assertFalse(results.isEmpty());

        System.out.println("✅ 在知识库 " + KNOWLEDGE_1 + " (分区 " + partition + ") 中搜索，返回 " + results.size() + " 条:");
        results.forEach(r -> {
            DocumentSegment seg = DocumentSegment.fromDocument(r.getDocument());
            System.out.println("   - " + seg.getId() + " [" + seg.getFileId() + "] (score: " + String.format("%.4f", r.getScore()) + ")");
        });
    }

    @Test
    @Order(31)
    @DisplayName("4.2 在多个知识库（分区）中搜索")
    void testSearchInMultiplePartitions() {
        List<Float> queryVector = createRandomVector(DIMENSION);
        List<String> partitions = DocumentSegment.getPartitionNames(Arrays.asList(KNOWLEDGE_1, KNOWLEDGE_2));

        List<SearchResult> results = vectorStore.similaritySearchInPartitions(queryVector, 10, partitions);

        assertFalse(results.isEmpty());

        System.out.println("✅ 在多个知识库中搜索，返回 " + results.size() + " 条:");
        results.forEach(r -> {
            DocumentSegment seg = DocumentSegment.fromDocument(r.getDocument());
            System.out.println("   - " + seg.getId() + " [" + seg.getFileId() + "] (score: " + String.format("%.4f", r.getScore()) + ")");
        });
    }

    @Test
    @Order(32)
    @DisplayName("4.3 在分区中带过滤条件搜索（指定文档）")
    void testSearchInPartitionWithFilter() {
        List<Float> queryVector = createRandomVector(DIMENSION);
        String partition = DocumentSegment.getPartitionName(KNOWLEDGE_1);
        String filter = DocumentSegment.filterByFileId(FILE_1);

        var request = com.fyj.rag.vectorstore.SearchRequest.builder()
                .vector(queryVector)
                .topK(3)
                .filter(filter)
                .build();

        List<SearchResult> results = vectorStore.similaritySearchInPartition(request, partition);

        assertFalse(results.isEmpty());

        results.forEach(r -> {
            DocumentSegment seg = DocumentSegment.fromDocument(r.getDocument());
            assertEquals(FILE_1, seg.getFileId());
        });

        System.out.println("✅ 在分区 " + partition + " 中搜索文档 " + FILE_1 + "，返回 " + results.size() + " 条");
    }

    // ==================== 5. 分区数据管理 ====================

    @Test
    @Order(40)
    @DisplayName("5.1 在指定分区中 Upsert")
    void testUpsertInPartition() {
        String partition = DocumentSegment.getPartitionName(KNOWLEDGE_1);

        DocumentSegment segment = DocumentSegment.builder()
                .id("upsert_test")
                .fileId(FILE_1)
                .content("在分区中 Upsert 的片段")
                .embedding(createRandomVector(DIMENSION))
                .build();

        vectorStore.upsert(Collections.singletonList(segment), partition);

        List<Document> results = vectorStore.getById(Collections.singletonList("upsert_test"), partition);
        assertFalse(results.isEmpty());

        System.out.println("✅ 在分区 " + partition + " 中 Upsert 成功");
    }

    @Test
    @Order(41)
    @DisplayName("5.2 从指定分区删除数据")
    void testDeleteInPartition() {
        String partition = DocumentSegment.getPartitionName(KNOWLEDGE_1);

        vectorStore.delete(Collections.singletonList("upsert_test"), partition);

        List<Document> results = vectorStore.getById(Collections.singletonList("upsert_test"), partition);
        assertTrue(results.isEmpty());

        System.out.println("✅ 从分区 " + partition + " 删除成功");
    }

    @Test
    @Order(42)
    @DisplayName("5.3 删除某文档的所有片段")
    void testDeleteByFileId() {
        String partition = DocumentSegment.getPartitionName(KNOWLEDGE_1);

        // 先插入测试数据
        String testFileId = "file_to_delete";
        List<DocumentSegment> segments = createTestSegments(testFileId, 0, 3);
        vectorStore.add(new ArrayList<>(segments), partition);

        long countBefore = vectorStore.count(partition);
        System.out.println("   删除前分区数据量: " + countBefore);

        // 根据文档ID删除
        String filter = DocumentSegment.filterByFileId(testFileId);
        vectorStore.deleteByFilter(filter, partition);

        long countAfter = vectorStore.count(partition);
        System.out.println("   删除后分区数据量: " + countAfter);

        assertEquals(countBefore - 3, countAfter);
        System.out.println("✅ 删除文档 " + testFileId + " 的所有片段成功");
    }

    // ==================== 6. 分区管理（知识库管理） ====================

    @Test
    @Order(50)
    @DisplayName("6.1 创建新知识库（新分区）")
    void testCreateNewKnowledgePartition() {
        String newKnowledgeId = "kb003";
        String partition = DocumentSegment.getPartitionName(newKnowledgeId);

        vectorStore.createPartition(partition);
        assertTrue(vectorStore.hasPartition(partition));

        List<DocumentSegment> segments = createTestSegments("new_file", 0, 2);
        vectorStore.add(new ArrayList<>(segments), partition);

        System.out.println("✅ 创建新知识库分区: " + partition);
        System.out.println("   当前所有分区: " + vectorStore.listPartitions());
    }

    @Test
    @Order(51)
    @DisplayName("6.2 删除知识库（删除分区及其数据）")
    void testDeleteKnowledgePartition() {
        String knowledgeId = "kb003";
        String partition = DocumentSegment.getPartitionName(knowledgeId);

        vectorStore.releasePartition(partition);
        vectorStore.dropPartition(partition);

        assertFalse(vectorStore.hasPartition(partition));
        System.out.println("✅ 删除知识库分区: " + partition + " (数据同时删除)");
    }

    // ==================== 7. 清理 ====================

    @Test
    @Order(100)
    @DisplayName("7.1 清理 - 删除 Collection")
    void testCleanup() {
        milvusClient.releaseCollection(DocumentSegment.COLLECTION_NAME);
        milvusClient.dropCollection(DocumentSegment.COLLECTION_NAME);

        assertFalse(milvusClient.hasCollection(DocumentSegment.COLLECTION_NAME));
        System.out.println("✅ Collection 已删除");
    }

    // ==================== 辅助方法 ====================

    private List<DocumentSegment> createTestSegments(String fileId, int startIndex, int count) {
        return IntStream.range(startIndex, startIndex + count)
                .mapToObj(i -> DocumentSegment.builder()
                        .id(fileId + "_" + i)
                        .fileId(fileId)
                        .content("文档[" + fileId + "]的第" + i + "个片段内容。这是测试文本。")
                        .embedding(createRandomVector(DIMENSION))
                        .metadata(Map.of("chunk_index", i, "total_chunks", count))
                        .build())
                .collect(Collectors.toList());
    }

    private List<Float> createRandomVector(int dimension) {
        Random random = new Random();
        List<Float> vector = new ArrayList<>(dimension);
        for (int i = 0; i < dimension; i++) {
            vector.add(random.nextFloat());
        }
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

