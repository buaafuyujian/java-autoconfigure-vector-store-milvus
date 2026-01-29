package com.example.demo;

import com.example.demo.entity.DocumentSegment;
import com.example.demo.repository.DocumentSegmentRepository;
import com.example.demo.repository.DocumentSegmentRepository.DocumentSegmentSearchResult;
import com.fyj.rag.client.MilvusClient;
import org.junit.jupiter.api.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * DocumentSegment 测试用例
 */
@SpringBootTest
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class DocumentSegmentTests {

    @Autowired
    private MilvusClient milvusClient;

    private static DocumentSegmentRepository repository;

    private static final int DIMENSION = 128;  // 测试用较小维度
    private static final String KNOWLEDGE_1 = "knowledge_001";
    private static final String KNOWLEDGE_2 = "knowledge_002";
    private static final String FILE_1 = "file_001";
    private static final String FILE_2 = "file_002";
    private static final String FILE_3 = "file_003";

    @BeforeAll
    static void setup(@Autowired MilvusClient client) {
        repository = new DocumentSegmentRepository(client, DIMENSION);
    }

    // ==================== Collection 初始化 ====================

    @Test
    @Order(1)
    @DisplayName("初始化 DocumentSegment Collection")
    void testInitCollection() {
        // 先删除旧的（如果存在）
        repository.dropCollection();

        // 初始化
        repository.initCollection();

        assertTrue(milvusClient.hasCollection(DocumentSegment.COLLECTION_NAME));
        System.out.println("✅ Collection initialized: " + DocumentSegment.COLLECTION_NAME);

        // 查看字段
        var desc = milvusClient.describeCollection(DocumentSegment.COLLECTION_NAME);
        System.out.println("   Fields: " + desc.getFieldNames());
    }

    // ==================== 插入操作测试 ====================

    @Test
    @Order(10)
    @DisplayName("测试插入单个文档片段")
    void testInsertSingle() {
        DocumentSegment segment = DocumentSegment.builder()
                .id("seg_001")
                .knowledgeId(KNOWLEDGE_1)
                .fileId(FILE_1)
                .content("这是第一个测试文档片段的内容")
                .embedding(createRandomVector(DIMENSION))
                .metadata(Map.of("chunk_index", 0, "total_chunks", 5))
                .build();

        repository.insert(segment);
        System.out.println("✅ Inserted single segment: " + segment.getId());
    }

    @Test
    @Order(11)
    @DisplayName("测试批量插入文档片段")
    void testInsertBatch() {
        // 知识库1，文档1 的片段
        List<DocumentSegment> segments1 = createTestSegments(KNOWLEDGE_1, FILE_1, 1, 5);
        repository.insert(segments1);
        System.out.println("✅ Inserted " + segments1.size() + " segments for " + KNOWLEDGE_1 + "/" + FILE_1);

        // 知识库1，文档2 的片段
        List<DocumentSegment> segments2 = createTestSegments(KNOWLEDGE_1, FILE_2, 0, 3);
        repository.insert(segments2);
        System.out.println("✅ Inserted " + segments2.size() + " segments for " + KNOWLEDGE_1 + "/" + FILE_2);

        // 知识库2，文档3 的片段
        List<DocumentSegment> segments3 = createTestSegments(KNOWLEDGE_2, FILE_3, 0, 4);
        repository.insert(segments3);
        System.out.println("✅ Inserted " + segments3.size() + " segments for " + KNOWLEDGE_2 + "/" + FILE_3);
    }

    // ==================== 查询操作测试 ====================

    @Test
    @Order(20)
    @DisplayName("测试根据ID查询")
    void testFindById() {
        Optional<DocumentSegment> result = repository.findById("seg_001");

        assertTrue(result.isPresent());
        assertEquals("seg_001", result.get().getId());
        assertEquals(KNOWLEDGE_1, result.get().getKnowledgeId());
        assertEquals(FILE_1, result.get().getFileId());
        System.out.println("✅ Found by ID: " + result.get().getId());
        System.out.println("   Content: " + result.get().getContent());
        System.out.println("   Metadata: " + result.get().getMetadata());
    }

    @Test
    @Order(21)
    @DisplayName("测试根据ID列表查询")
    void testFindByIds() {
        List<DocumentSegment> results = repository.findByIds(Arrays.asList(
                "seg_001",
                KNOWLEDGE_1 + "_" + FILE_1 + "_1",
                KNOWLEDGE_1 + "_" + FILE_1 + "_2"
        ));

        assertFalse(results.isEmpty());
        System.out.println("✅ Found by IDs: " + results.size());
        results.forEach(s -> System.out.println("   - " + s.getId() + ": " + s.getContent().substring(0, Math.min(30, s.getContent().length())) + "..."));
    }

    @Test
    @Order(22)
    @DisplayName("测试根据知识库ID查询")
    void testFindByKnowledgeId() {
        List<DocumentSegment> results = repository.findByKnowledgeId(KNOWLEDGE_1);

        assertFalse(results.isEmpty());
        results.forEach(s -> assertEquals(KNOWLEDGE_1, s.getKnowledgeId()));
        System.out.println("✅ Found " + results.size() + " segments in knowledge: " + KNOWLEDGE_1);
    }

    @Test
    @Order(23)
    @DisplayName("测试根据文档ID查询")
    void testFindByFileId() {
        List<DocumentSegment> results = repository.findByFileId(FILE_1);

        assertFalse(results.isEmpty());
        results.forEach(s -> assertEquals(FILE_1, s.getFileId()));
        System.out.println("✅ Found " + results.size() + " segments in file: " + FILE_1);
    }

    @Test
    @Order(24)
    @DisplayName("测试根据知识库ID和文档ID查询")
    void testFindByKnowledgeIdAndFileId() {
        List<DocumentSegment> results = repository.findByKnowledgeIdAndFileId(KNOWLEDGE_1, FILE_2);

        assertFalse(results.isEmpty());
        results.forEach(s -> {
            assertEquals(KNOWLEDGE_1, s.getKnowledgeId());
            assertEquals(FILE_2, s.getFileId());
        });
        System.out.println("✅ Found " + results.size() + " segments in " + KNOWLEDGE_1 + "/" + FILE_2);
    }

    @Test
    @Order(25)
    @DisplayName("测试统计数量")
    void testCount() {
        long countByKnowledge1 = repository.countByKnowledgeId(KNOWLEDGE_1);
        long countByKnowledge2 = repository.countByKnowledgeId(KNOWLEDGE_2);
        long countByFile1 = repository.countByFileId(FILE_1);

        System.out.println("✅ Count by knowledge " + KNOWLEDGE_1 + ": " + countByKnowledge1);
        System.out.println("   Count by knowledge " + KNOWLEDGE_2 + ": " + countByKnowledge2);
        System.out.println("   Count by file " + FILE_1 + ": " + countByFile1);

        assertTrue(countByKnowledge1 > 0);
        assertTrue(countByKnowledge2 > 0);
    }

    // ==================== 向量搜索测试 ====================

    @Test
    @Order(30)
    @DisplayName("测试全局向量搜索")
    void testSearchAll() {
        List<Float> queryVector = createRandomVector(DIMENSION);

        List<DocumentSegmentSearchResult> results = repository.search(queryVector, 5);

        assertFalse(results.isEmpty());
        System.out.println("✅ Global search results: " + results.size());
        results.forEach(r -> System.out.println("   - " + r.getSegment().getId()
                + " [" + r.getSegment().getKnowledgeId() + "/" + r.getSegment().getFileId() + "]"
                + " (score: " + r.getScore() + ")"));
    }

    @Test
    @Order(31)
    @DisplayName("测试在指定知识库中搜索")
    void testSearchInKnowledge() {
        List<Float> queryVector = createRandomVector(DIMENSION);

        List<DocumentSegmentSearchResult> results = repository.searchInKnowledge(queryVector, 5, KNOWLEDGE_1);

        assertFalse(results.isEmpty());
        results.forEach(r -> assertEquals(KNOWLEDGE_1, r.getSegment().getKnowledgeId()));
        System.out.println("✅ Search in knowledge " + KNOWLEDGE_1 + ": " + results.size());
        results.forEach(r -> System.out.println("   - " + r.getSegment().getId() + " (score: " + r.getScore() + ")"));
    }

    @Test
    @Order(32)
    @DisplayName("测试在指定文档中搜索")
    void testSearchInFile() {
        List<Float> queryVector = createRandomVector(DIMENSION);

        List<DocumentSegmentSearchResult> results = repository.searchInFile(queryVector, 3, FILE_1);

        assertFalse(results.isEmpty());
        results.forEach(r -> assertEquals(FILE_1, r.getSegment().getFileId()));
        System.out.println("✅ Search in file " + FILE_1 + ": " + results.size());
        results.forEach(r -> System.out.println("   - " + r.getSegment().getId() + " (score: " + r.getScore() + ")"));
    }

    @Test
    @Order(33)
    @DisplayName("测试在多个知识库中搜索")
    void testSearchInMultipleKnowledges() {
        List<Float> queryVector = createRandomVector(DIMENSION);

        List<DocumentSegmentSearchResult> results = repository.searchInKnowledges(
                queryVector, 10, Arrays.asList(KNOWLEDGE_1, KNOWLEDGE_2));

        assertFalse(results.isEmpty());
        System.out.println("✅ Search in multiple knowledges: " + results.size());
        results.forEach(r -> System.out.println("   - " + r.getSegment().getId()
                + " [" + r.getSegment().getKnowledgeId() + "]"
                + " (score: " + r.getScore() + ")"));
    }

    // ==================== Upsert 测试 ====================

    @Test
    @Order(40)
    @DisplayName("测试 Upsert 操作")
    void testUpsert() {
        // 更新已存在的
        DocumentSegment updateSegment = DocumentSegment.builder()
                .id("seg_001")
                .knowledgeId(KNOWLEDGE_1)
                .fileId(FILE_1)
                .content("这是更新后的内容 - Updated content")
                .embedding(createRandomVector(DIMENSION))
                .metadata(Map.of("chunk_index", 0, "total_chunks", 5, "updated", true))
                .build();

        // 新增的
        DocumentSegment newSegment = DocumentSegment.builder()
                .id("seg_upsert_new")
                .knowledgeId(KNOWLEDGE_1)
                .fileId(FILE_1)
                .content("通过 upsert 新增的片段")
                .embedding(createRandomVector(DIMENSION))
                .metadata(Map.of("source", "upsert"))
                .build();

        repository.upsert(Arrays.asList(updateSegment, newSegment));

        // 验证
        Optional<DocumentSegment> updated = repository.findById("seg_001");
        assertTrue(updated.isPresent());
        assertTrue(updated.get().getContent().contains("Updated"));
        System.out.println("✅ Upsert completed");
        System.out.println("   Updated content: " + updated.get().getContent());
    }

    // ==================== 删除测试 ====================

    @Test
    @Order(50)
    @DisplayName("测试根据ID删除")
    void testDeleteById() {
        // 先添加一个测试片段
        DocumentSegment segment = DocumentSegment.builder()
                .id("to_delete_001")
                .knowledgeId(KNOWLEDGE_1)
                .fileId(FILE_1)
                .content("This will be deleted")
                .embedding(createRandomVector(DIMENSION))
                .build();
        repository.insert(segment);

        // 删除
        repository.deleteById("to_delete_001");

        // 验证
        Optional<DocumentSegment> result = repository.findById("to_delete_001");
        assertTrue(result.isEmpty());
        System.out.println("✅ Deleted by ID");
    }

    @Test
    @Order(51)
    @DisplayName("测试根据文档ID删除所有片段")
    void testDeleteByFileId() {
        // 先添加测试片段
        List<DocumentSegment> segments = createTestSegments("temp_knowledge", "temp_file", 0, 3);
        repository.insert(segments);

        long countBefore = repository.countByFileId("temp_file");
        System.out.println("   Before delete: " + countBefore + " segments");

        // 删除
        repository.deleteByFileId("temp_file");

        long countAfter = repository.countByFileId("temp_file");
        assertEquals(0, countAfter);
        System.out.println("✅ Deleted all segments in file: temp_file");
    }

    // ==================== 清理测试 ====================

    @Test
    @Order(100)
    @DisplayName("清理测试数据 - 删除 Collection")
    void testCleanup() {
        repository.dropCollection();

        assertFalse(milvusClient.hasCollection(DocumentSegment.COLLECTION_NAME));
        System.out.println("✅ Collection dropped: " + DocumentSegment.COLLECTION_NAME);
    }

    // ==================== 辅助方法 ====================

    /**
     * 创建测试片段
     */
    private List<DocumentSegment> createTestSegments(String knowledgeId, String fileId, int startIndex, int count) {
        return IntStream.range(startIndex, startIndex + count)
                .mapToObj(i -> DocumentSegment.builder()
                        .id(knowledgeId + "_" + fileId + "_" + i)
                        .knowledgeId(knowledgeId)
                        .fileId(fileId)
                        .content("这是知识库[" + knowledgeId + "]文档[" + fileId + "]的第" + i + "个片段的内容。" +
                                "Lorem ipsum dolor sit amet, consectetur adipiscing elit.")
                        .embedding(createRandomVector(DIMENSION))
                        .metadata(Map.of(
                                "chunk_index", i,
                                "total_chunks", count,
                                "create_time", System.currentTimeMillis()
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

