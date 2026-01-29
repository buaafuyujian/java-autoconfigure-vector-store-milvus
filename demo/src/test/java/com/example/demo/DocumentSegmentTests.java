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
 * DocumentSegment 测试用例 - 直接使用 MilvusVectorStore
 * <p>
 * 由于 DocumentSegment 继承 Document，可以直接使用 MilvusVectorStore 的所有方法
 * 不需要额外的 Repository 层
 * <p>
 * 注意：运行测试前需要启动 Milvus 服务
 */
@SpringBootTest
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class DocumentSegmentTests {

    @Autowired
    private MilvusClient milvusClient;

    private static MilvusVectorStore vectorStore;

    private static final int DIMENSION = 128;
    private static final String KNOWLEDGE_1 = "knowledge_001";
    private static final String KNOWLEDGE_2 = "knowledge_002";
    private static final String FILE_1 = "file_001";
    private static final String FILE_2 = "file_002";
    private static final String FILE_3 = "file_003";

    @BeforeAll
    static void setup(@Autowired MilvusClient client) {
        // 直接获取 VectorStore，指定额外输出字段
        vectorStore = client.getVectorStore(
                DocumentSegment.COLLECTION_NAME,
                DocumentSegment.FIELD_ID,
                DocumentSegment.FIELD_CONTENT,
                DocumentSegment.FIELD_EMBEDDING,
                DocumentSegment.FIELD_METADATA,
                Arrays.asList(DocumentSegment.FIELD_KNOWLEDGE_ID, DocumentSegment.FIELD_FILE_ID)
        );
    }

    // ==================== 1. Collection 初始化 ====================

    @Test
    @Order(1)
    @DisplayName("1.1 初始化 Collection")
    void testInitCollection() {
        // 删除旧的
        if (milvusClient.hasCollection(DocumentSegment.COLLECTION_NAME)) {
            milvusClient.releaseCollection(DocumentSegment.COLLECTION_NAME);
            milvusClient.dropCollection(DocumentSegment.COLLECTION_NAME);
        }

        // 创建新的
        milvusClient.createCollection(
                DocumentSegment.COLLECTION_NAME,
                DocumentSegment.createSchema(DIMENSION),
                DocumentSegment.createIndex()
        );
        milvusClient.loadCollection(DocumentSegment.COLLECTION_NAME);

        assertTrue(milvusClient.hasCollection(DocumentSegment.COLLECTION_NAME));
        System.out.println("✅ Collection 创建成功: " + DocumentSegment.COLLECTION_NAME);
    }

    // ==================== 2. 插入操作 ====================

    @Test
    @Order(10)
    @DisplayName("2.1 插入单个 DocumentSegment")
    void testInsertSingle() {
        DocumentSegment segment = DocumentSegment.builder()
                .id("seg_001")
                .knowledgeId(KNOWLEDGE_1)
                .fileId(FILE_1)
                .content("这是第一个测试文档片段")
                .embedding(createRandomVector(DIMENSION))
                .metadata(Map.of("chunk_index", 0))
                .build();

        // 直接使用 vectorStore.add()
        // DocumentSegment 继承 Document，可以直接放入 List<Document>
        List<Document> docs = Collections.singletonList(segment);
        vectorStore.add(docs);

        System.out.println("✅ 插入成功: " + segment.getId());
    }

    @Test
    @Order(11)
    @DisplayName("2.2 批量插入 DocumentSegment")
    void testInsertBatch() {
        // 知识库1，文档1
        List<DocumentSegment> segments1 = createTestSegments(KNOWLEDGE_1, FILE_1, 1, 5);
        vectorStore.add(new ArrayList<>(segments1));  // DocumentSegment 可以直接转为 List<Document>
        System.out.println("✅ 插入 " + segments1.size() + " 个片段到 " + KNOWLEDGE_1 + "/" + FILE_1);

        // 知识库1，文档2
        List<DocumentSegment> segments2 = createTestSegments(KNOWLEDGE_1, FILE_2, 0, 3);
        vectorStore.add(new ArrayList<>(segments2));
        System.out.println("✅ 插入 " + segments2.size() + " 个片段到 " + KNOWLEDGE_1 + "/" + FILE_2);

        // 知识库2，文档3
        List<DocumentSegment> segments3 = createTestSegments(KNOWLEDGE_2, FILE_3, 0, 4);
        vectorStore.add(new ArrayList<>(segments3));
        System.out.println("✅ 插入 " + segments3.size() + " 个片段到 " + KNOWLEDGE_2 + "/" + FILE_3);
    }

    // ==================== 3. 查询操作 ====================

    @Test
    @Order(20)
    @DisplayName("3.1 根据 ID 查询")
    void testGetById() {
        List<Document> results = vectorStore.getById(Collections.singletonList("seg_001"));

        assertFalse(results.isEmpty());

        // 转换为 DocumentSegment
        DocumentSegment segment = DocumentSegment.fromDocument(results.get(0));

        assertEquals("seg_001", segment.getId());
        assertEquals(KNOWLEDGE_1, segment.getKnowledgeId());
        assertEquals(FILE_1, segment.getFileId());

        System.out.println("✅ 查询成功: " + segment.getId());
        System.out.println("   knowledgeId: " + segment.getKnowledgeId());
        System.out.println("   fileId: " + segment.getFileId());
        System.out.println("   content: " + segment.getContent());
    }

    @Test
    @Order(21)
    @DisplayName("3.2 根据知识库ID查询")
    void testQueryByKnowledgeId() {
        // 使用 DocumentSegment 提供的过滤表达式
        String filter = DocumentSegment.filterByKnowledgeId(KNOWLEDGE_1);
        List<Document> results = vectorStore.query(filter, 100);

        assertFalse(results.isEmpty());

        // 转换并验证
        List<DocumentSegment> segments = results.stream()
                .map(DocumentSegment::fromDocument)
                .toList();

        segments.forEach(s -> assertEquals(KNOWLEDGE_1, s.getKnowledgeId()));

        System.out.println("✅ 知识库 " + KNOWLEDGE_1 + " 有 " + segments.size() + " 个片段");
    }

    @Test
    @Order(22)
    @DisplayName("3.3 根据文档ID查询")
    void testQueryByFileId() {
        String filter = DocumentSegment.filterByFileId(FILE_1);
        List<Document> results = vectorStore.query(filter, 100);

        assertFalse(results.isEmpty());

        List<DocumentSegment> segments = results.stream()
                .map(DocumentSegment::fromDocument)
                .toList();

        segments.forEach(s -> assertEquals(FILE_1, s.getFileId()));

        System.out.println("✅ 文档 " + FILE_1 + " 有 " + segments.size() + " 个片段");
    }

    @Test
    @Order(23)
    @DisplayName("3.4 组合条件查询")
    void testQueryByKnowledgeIdAndFileId() {
        String filter = DocumentSegment.filterByKnowledgeIdAndFileId(KNOWLEDGE_1, FILE_2);
        List<Document> results = vectorStore.query(filter, 100);

        assertFalse(results.isEmpty());

        List<DocumentSegment> segments = results.stream()
                .map(DocumentSegment::fromDocument)
                .toList();

        segments.forEach(s -> {
            assertEquals(KNOWLEDGE_1, s.getKnowledgeId());
            assertEquals(FILE_2, s.getFileId());
        });

        System.out.println("✅ " + KNOWLEDGE_1 + "/" + FILE_2 + " 有 " + segments.size() + " 个片段");
    }

    // ==================== 4. 向量搜索 ====================

    @Test
    @Order(30)
    @DisplayName("4.1 全局向量搜索")
    void testGlobalSearch() {
        List<Float> queryVector = createRandomVector(DIMENSION);

        List<SearchResult> results = vectorStore.similaritySearch(queryVector, 5);

        assertFalse(results.isEmpty());
        System.out.println("✅ 全局搜索返回 " + results.size() + " 条结果:");

        results.forEach(r -> {
            DocumentSegment seg = DocumentSegment.fromDocument(r.getDocument());
            System.out.println("   - " + seg.getId() +
                    " [" + seg.getKnowledgeId() + "/" + seg.getFileId() + "]" +
                    " (score: " + String.format("%.4f", r.getScore()) + ")");
        });
    }

    @Test
    @Order(31)
    @DisplayName("4.2 在指定知识库中搜索")
    void testSearchInKnowledge() {
        List<Float> queryVector = createRandomVector(DIMENSION);
        String filter = DocumentSegment.filterByKnowledgeId(KNOWLEDGE_1);

        // 使用带过滤条件的搜索
        List<SearchResult> results = vectorStore.similaritySearch(queryVector, 5, filter);

        assertFalse(results.isEmpty());

        // 验证所有结果都属于指定知识库
        results.forEach(r -> {
            DocumentSegment seg = DocumentSegment.fromDocument(r.getDocument());
            assertEquals(KNOWLEDGE_1, seg.getKnowledgeId());
        });

        System.out.println("✅ 在知识库 " + KNOWLEDGE_1 + " 中搜索，返回 " + results.size() + " 条结果");
    }

    @Test
    @Order(32)
    @DisplayName("4.3 在多个知识库中搜索")
    void testSearchInMultipleKnowledges() {
        List<Float> queryVector = createRandomVector(DIMENSION);
        String filter = DocumentSegment.filterByKnowledgeIds(Arrays.asList(KNOWLEDGE_1, KNOWLEDGE_2));

        List<SearchResult> results = vectorStore.similaritySearch(queryVector, 10, filter);

        assertFalse(results.isEmpty());
        System.out.println("✅ 在多个知识库中搜索，返回 " + results.size() + " 条结果:");

        results.forEach(r -> {
            DocumentSegment seg = DocumentSegment.fromDocument(r.getDocument());
            System.out.println("   - " + seg.getId() + " [" + seg.getKnowledgeId() + "]");
        });
    }

    @Test
    @Order(33)
    @DisplayName("4.4 在指定文档中搜索")
    void testSearchInFile() {
        List<Float> queryVector = createRandomVector(DIMENSION);
        String filter = DocumentSegment.filterByFileId(FILE_1);

        List<SearchResult> results = vectorStore.similaritySearch(queryVector, 3, filter);

        assertFalse(results.isEmpty());

        results.forEach(r -> {
            DocumentSegment seg = DocumentSegment.fromDocument(r.getDocument());
            assertEquals(FILE_1, seg.getFileId());
        });

        System.out.println("✅ 在文档 " + FILE_1 + " 中搜索，返回 " + results.size() + " 条结果");
    }

    // ==================== 5. Upsert ====================

    @Test
    @Order(40)
    @DisplayName("5.1 Upsert 操作")
    void testUpsert() {
        // 更新已存在的
        DocumentSegment updateSegment = DocumentSegment.builder()
                .id("seg_001")
                .knowledgeId(KNOWLEDGE_1)
                .fileId(FILE_1)
                .content("【已更新】Updated content via upsert")
                .embedding(createRandomVector(DIMENSION))
                .metadata(Map.of("updated", true))
                .build();

        // 新增的
        DocumentSegment newSegment = DocumentSegment.builder()
                .id("upsert_new")
                .knowledgeId(KNOWLEDGE_1)
                .fileId(FILE_1)
                .content("通过 upsert 新增的片段")
                .embedding(createRandomVector(DIMENSION))
                .build();

        vectorStore.upsert(Arrays.asList(updateSegment, newSegment));

        // 验证
        List<Document> results = vectorStore.getById(Arrays.asList("seg_001", "upsert_new"));
        assertEquals(2, results.size());

        System.out.println("✅ Upsert 成功");
    }

    // ==================== 6. 删除 ====================

    @Test
    @Order(50)
    @DisplayName("6.1 根据 ID 删除")
    void testDeleteById() {
        // 先插入
        DocumentSegment segment = DocumentSegment.builder()
                .id("to_delete")
                .knowledgeId(KNOWLEDGE_1)
                .fileId(FILE_1)
                .content("将被删除")
                .embedding(createRandomVector(DIMENSION))
                .build();
        vectorStore.add(Collections.singletonList(segment));

        // 删除
        vectorStore.delete(Collections.singletonList("to_delete"));

        // 验证
        List<Document> results = vectorStore.getById(Collections.singletonList("to_delete"));
        assertTrue(results.isEmpty());

        System.out.println("✅ 删除成功");
    }

    @Test
    @Order(51)
    @DisplayName("6.2 根据过滤条件删除（删除整个文档的片段）")
    void testDeleteByFilter() {
        // 先插入测试数据
        String testFileId = "file_to_delete";
        List<DocumentSegment> segments = createTestSegments("temp_kb", testFileId, 0, 3);
        vectorStore.add(new ArrayList<>(segments));

        // 根据文档ID删除
        String filter = DocumentSegment.filterByFileId(testFileId);
        vectorStore.deleteByFilter(filter);

        // 验证
        List<Document> results = vectorStore.query(filter, 100);
        assertTrue(results.isEmpty());

        System.out.println("✅ 根据文档ID删除所有片段成功");
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

    private List<DocumentSegment> createTestSegments(String knowledgeId, String fileId, int startIndex, int count) {
        return IntStream.range(startIndex, startIndex + count)
                .mapToObj(i -> DocumentSegment.builder()
                        .id(knowledgeId + "_" + fileId + "_" + i)
                        .knowledgeId(knowledgeId)
                        .fileId(fileId)
                        .content("知识库[" + knowledgeId + "]文档[" + fileId + "]的第" + i + "个片段内容")
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

