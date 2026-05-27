package com.example.demo;

import com.example.demo.entity.DocumentSegment;
import com.fyj.rag.client.MilvusClient;
import com.fyj.rag.schema.IndexSchema;
import com.fyj.rag.vectorstore.MilvusVectorStore;
import com.fyj.rag.vectorstore.model.Document;
import com.fyj.rag.vectorstore.model.SearchResult;
import com.fyj.rag.vectorstore.request.BatchSearchRequest;
import com.fyj.rag.vectorstore.request.QueryRequest;
import com.fyj.rag.vectorstore.request.SearchRequest;
import com.fyj.rag.vectorstore.request.SearchType;
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
 * 3. 支持向量搜索、BM25 全文检索、混合搜索三种搜索模式
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
    @DisplayName("1.1 初始化 Collection（支持 BM25）")
    void testInitCollection() {
        if (milvusClient.hasCollection(DocumentSegment.COLLECTION_NAME)) {
            milvusClient.releaseCollection(DocumentSegment.COLLECTION_NAME);
            milvusClient.dropCollection(DocumentSegment.COLLECTION_NAME);
        }

        // 使用 EmbeddingModel 的维度创建支持 BM25 的 Schema
        int dimension = embeddingModel.dimensions();
        System.out.println("   EmbeddingModel 维度: " + dimension);

        // 使用支持 BM25 的 Schema（包含 BM25 Function）
        List<IndexSchema> indexes = DocumentSegment.createAllIndexes();
        milvusClient.createCollection(
                DocumentSegment.COLLECTION_NAME,
                DocumentSegment.createSchemaWithBM25(dimension),
                indexes
        );
        milvusClient.loadCollection(DocumentSegment.COLLECTION_NAME);

        assertTrue(milvusClient.hasCollection(DocumentSegment.COLLECTION_NAME));
        System.out.println("✅ Collection 创建成功（支持向量搜索 + BM25）: " + DocumentSegment.COLLECTION_NAME);
    }

    @Test
    @Order(2)
    @DisplayName("1.2 为每个知识库创建分区")
    void testCreatePartitions() {
        String partition1 = DocumentSegment.getPartitionName(KNOWLEDGE_1);
        vectorStore.createPartition(partition1);
        // 新分区在 Collection load 之后创建，需要显式 load 才能参与查询/统计
        vectorStore.loadPartition(partition1);
        System.out.println("✅ 创建并加载分区: " + partition1);

        String partition2 = DocumentSegment.getPartitionName(KNOWLEDGE_2);
        vectorStore.createPartition(partition2);
        vectorStore.loadPartition(partition2);
        System.out.println("✅ 创建并加载分区: " + partition2);

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
    void testCountByPartition() throws InterruptedException {
        String partition1 = DocumentSegment.getPartitionName(KNOWLEDGE_1);
        String partition2 = DocumentSegment.getPartitionName(KNOWLEDGE_2);

        // 插入后数据在 growing segment，count(*) 只扫描 sealed segment
        // flush 将 growing segment 封闭为 sealed segment，使数据立即可计
        vectorStore.flush();

        // 等待 Milvus 完成索引构建，避免搜索时数据尚未就绪
        Thread.sleep(2000);

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

        // 使用泛型 QueryRequest，类型信息在 Request 中
        QueryRequest<DocumentSegment> request = QueryRequest.<DocumentSegment>builder()
                .filter(filter)
                .partitionName(partition)
                .limit(100)
                .documentClass(DocumentSegment.class)
                .build();

        List<DocumentSegment> segments = vectorStore.query(request);

        assertFalse(segments.isEmpty());

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

        // 使用泛型 getById 方法直接返回 DocumentSegment 类型
        List<DocumentSegment> results = vectorStore.getById(Collections.singletonList(id), partition, DocumentSegment.class);

        assertFalse(results.isEmpty());
        DocumentSegment segment = results.get(0);

        assertEquals(id, segment.getId());
        assertEquals(FILE_1, segment.getFileId());

        System.out.println("✅ 获取成功: " + segment.getId());
        System.out.println("   fileId: " + segment.getFileId());
        System.out.println("   content: " + segment.getContent());
    }

    @Test
    @Order(22)
    @DisplayName("3.3 测试泛型 QueryRequest 用法")
    void testGenericQueryMethods() {
        String partition = DocumentSegment.getPartitionName(KNOWLEDGE_1);
        String filter = DocumentSegment.filterByFileId(FILE_1);

        // 方式1: 使用简单的便捷方法
        List<DocumentSegment> result1 = vectorStore.query(filter, DocumentSegment.class);
        System.out.println("✅ query(filter, class): 返回 " + result1.size() + " 条");

        // 方式2: 使用静态工厂方法
        QueryRequest<Document> request2 = QueryRequest.of(filter);
        List<Document> result2 = vectorStore.query(request2);
        System.out.println("✅ QueryRequest.of(filter): 返回 " + result2.size() + " 条");

        // 方式3: 使用 Builder 指定分区和类型
        QueryRequest<DocumentSegment> request3 = QueryRequest.<DocumentSegment>builder()
                .filter(filter)
                .partitionName(partition)
                .documentClass(DocumentSegment.class)
                .build();
        List<DocumentSegment> result3 = vectorStore.query(request3);
        System.out.println("✅ QueryRequest with partition: 返回 " + result3.size() + " 条");

        // 方式4: 使用 Builder 完整参数（分区+分页+类型）
        QueryRequest<DocumentSegment> request4 = QueryRequest.<DocumentSegment>builder()
                .filter(filter)
                .partitionName(partition)
                .offset(0)
                .limit(10)
                .documentClass(DocumentSegment.class)
                .build();
        List<DocumentSegment> result4 = vectorStore.query(request4);
        System.out.println("✅ QueryRequest with pagination: 返回 " + result4.size() + " 条");

        // 方式5: 使用 of 静态工厂方法（带分区）
        QueryRequest<Document> request5 = QueryRequest.of(filter, partition);
        List<Document> result5 = vectorStore.query(request5);
        System.out.println("✅ QueryRequest.of(filter, partition): 返回 " + result5.size() + " 条");

        // 验证所有结果都能直接获取 fileId
        assertFalse(result1.isEmpty());
        result1.forEach(s -> {
            assertNotNull(s.getFileId());
            assertEquals(FILE_1, s.getFileId());
        });
    }

    // ==================== 4. 文本搜索（泛型 SearchRequest）====================

    @Test
    @Order(30)
    @DisplayName("4.1 使用文本搜索（泛型 SearchRequest）")
    void testTextSearch() {
        String partition = DocumentSegment.getPartitionName(KNOWLEDGE_1);

        // 使用泛型 SearchRequest，类型信息在 Request 中
        String queryText = "Java 编程语言";
        SearchRequest<DocumentSegment> request = SearchRequest.<DocumentSegment>builder()
                .query(queryText)
                .topK(3)
                .inPartition(partition)
                .documentClass(DocumentSegment.class)
                .build();

        List<SearchResult<DocumentSegment>> results = vectorStore.search(request);

        assertFalse(results.isEmpty());

        System.out.println("✅ 文本搜索 \"" + queryText + "\"，返回 " + results.size() + " 条:");
        results.forEach(r -> {
            DocumentSegment seg = r.getDocument();
            System.out.println("   - " + seg.getId() + " (score: " + String.format("%.4f", r.getScore()) + ")");
            System.out.println("     content: " + seg.getContent());
            System.out.println("     fileId: " + seg.getFileId());
        });
    }

    @Test
    @Order(31)
    @DisplayName("4.2 在多个知识库中文本搜索")
    void testTextSearchInMultiplePartitions() {
        List<String> partitions = DocumentSegment.getPartitionNames(Arrays.asList(KNOWLEDGE_1, KNOWLEDGE_2));

        // 使用泛型 SearchRequest
        String queryText = "Spring Boot 框架";
        SearchRequest<DocumentSegment> request = SearchRequest.<DocumentSegment>builder()
                .query(queryText)
                .topK(5)
                .partitionNames(partitions)
                .documentClass(DocumentSegment.class)
                .build();

        List<SearchResult<DocumentSegment>> results = vectorStore.search(request);

        assertFalse(results.isEmpty());

        System.out.println("✅ 跨知识库搜索 \"" + queryText + "\"，返回 " + results.size() + " 条:");
        results.forEach(r -> {
            DocumentSegment seg = r.getDocument();
            System.out.println("   - " + seg.getId() + " [" + seg.getFileId() + "] (score: " + String.format("%.4f", r.getScore()) + ")");
            System.out.println("     content: " + seg.getContent());
        });
    }

    @Test
    @Order(32)
    @DisplayName("4.3 全局文本搜索")
    void testGlobalTextSearch() {
        // 使用泛型 SearchRequest
        String queryText = "人工智能技术";
        SearchRequest<DocumentSegment> request = SearchRequest.<DocumentSegment>builder()
                .query(queryText)
                .topK(5)
                .documentClass(DocumentSegment.class)
                .build();

        List<SearchResult<DocumentSegment>> results = vectorStore.search(request);

        assertFalse(results.isEmpty());

        System.out.println("✅ 全局搜索 \"" + queryText + "\"，返回 " + results.size() + " 条:");
        results.forEach(r -> {
            DocumentSegment seg = r.getDocument();
            System.out.println("   - " + seg.getId() + " [fileId: " + seg.getFileId() + "] (score: " + String.format("%.4f", r.getScore()) + ")");
            System.out.println("     content: " + seg.getContent());
        });
    }

    // ==================== 4.5 BM25 全文检索 ====================

    @Test
    @Order(33)
    @DisplayName("4.4 BM25 全文检索")
    void testBM25Search() {
        String partition = DocumentSegment.getPartitionName(KNOWLEDGE_1);

        // 使用 BM25 搜索类型
        String queryText = "框架";
        SearchRequest<DocumentSegment> request = SearchRequest.<DocumentSegment>builder()
                .query(queryText)
                .searchType(SearchType.BM25)
                .topK(5)
                .inPartition(partition)
                .documentClass(DocumentSegment.class)
                .build();

        List<SearchResult<DocumentSegment>> results = vectorStore.search(request);

        System.out.println("✅ BM25 搜索 \"" + queryText + "\"，返回 " + results.size() + " 条:");
        results.forEach(r -> {
            DocumentSegment seg = r.getDocument();
            System.out.println("   - " + seg.getId() + " (score: " + String.format("%.4f", r.getScore()) + ")");
            System.out.println("     content: " + seg.getContent());
            System.out.println("     fileId: " + seg.getFileId());
        });
    }

    @Test
    @Order(34)
    @DisplayName("4.5 BM25 搜索（使用便捷方法）")
    void testBM25SearchWithConvenienceMethod() {
        // 使用 SearchRequest.bm25() 便捷方法
        String queryText = "Spring Boot 框架";
        SearchRequest<Document> request = SearchRequest.bm25(queryText, 5);

        List<SearchResult<Document>> results = vectorStore.search(request);

        System.out.println("✅ BM25 便捷搜索 \"" + queryText + "\"，返回 " + results.size() + " 条:");
        results.forEach(r -> {
            Document doc = r.getDocument();
            System.out.println("   - " + doc.getId() + " (score: " + String.format("%.4f", r.getScore()) + ")");
            System.out.println("     content: " + doc.getContent());
        });
    }

    // ==================== 4.6 混合搜索（向量 + BM25）====================

    @Test
    @Order(35)
    @DisplayName("4.6 混合搜索（向量 + BM25）")
    void testHybridSearch() {
        String partition = DocumentSegment.getPartitionName(KNOWLEDGE_1);

        // 使用混合搜索，向量权重 0.7，BM25 权重 0.3
        String queryText = "人工智能 机器学习";
        SearchRequest<DocumentSegment> request = SearchRequest.<DocumentSegment>builder()
                .query(queryText)
                .searchType(SearchType.HYBRID)
                .vectorWeight(0.7f)
                .bm25Weight(0.3f)
                .topK(5)
                .inPartition(partition)
                .documentClass(DocumentSegment.class)
                .build();

        List<SearchResult<DocumentSegment>> results = vectorStore.search(request);

        System.out.println("✅ 混合搜索 \"" + queryText + "\"（向量:70% + BM25:30%），返回 " + results.size() + " 条:");
        results.forEach(r -> {
            DocumentSegment seg = r.getDocument();
            System.out.println("   - " + seg.getId() + " (score: " + String.format("%.4f", r.getScore()) + ")");
            System.out.println("     content: " + seg.getContent());
            System.out.println("     fileId: " + seg.getFileId());
        });
    }

    @Test
    @Order(36)
    @DisplayName("4.7 混合搜索（使用便捷方法）")
    void testHybridSearchWithConvenienceMethod() {
        // 使用 SearchRequest.hybrid() 便捷方法（默认权重各 50%）
        String queryText = "深度学习 图像识别";
        SearchRequest<Document> request = SearchRequest.hybrid(queryText, 5);

        List<SearchResult<Document>> results = vectorStore.search(request);

        System.out.println("✅ 混合便捷搜索 \"" + queryText + "\"（默认各50%），返回 " + results.size() + " 条:");
        results.forEach(r -> {
            Document doc = r.getDocument();
            System.out.println("   - " + doc.getId() + " (score: " + String.format("%.4f", r.getScore()) + ")");
            System.out.println("     content: " + doc.getContent());
        });
    }

    @Test
    @Order(37)
    @DisplayName("4.8 混合搜索（自定义权重）")
    void testHybridSearchWithCustomWeights() {
        List<String> partitions = DocumentSegment.getPartitionNames(Arrays.asList(KNOWLEDGE_1, KNOWLEDGE_2));

        // 使用自定义权重的混合搜索（向量 30% + BM25 70%）
        String queryText = "编程语言 框架";
        SearchRequest<Document> request = SearchRequest.hybrid(queryText, 5, 0.3f, 0.7f);

        List<SearchResult<Document>> results = vectorStore.search(request);

        System.out.println("✅ 混合搜索 \"" + queryText + "\"（向量:30% + BM25:70%），返回 " + results.size() + " 条:");
        results.forEach(r -> {
            Document doc = r.getDocument();
            System.out.println("   - " + doc.getId() + " (score: " + String.format("%.4f", r.getScore()) + ")");
            System.out.println("     content: " + doc.getContent());
        });
    }

    @Test
    @Order(38)
    @DisplayName("4.9 对比三种搜索方式的结果")
    void testCompareSearchMethods() {
        String queryText = "人工智能";
        int topK = 3;

        System.out.println("✅ 对比三种搜索方式（查询: \"" + queryText + "\"）:");
        System.out.println();

        // 1. 向量搜索
        SearchRequest<DocumentSegment> vectorReq = SearchRequest.<DocumentSegment>builder()
                .query(queryText)
                .searchType(SearchType.VECTOR)
                .topK(topK)
                .documentClass(DocumentSegment.class)
                .build();
        List<SearchResult<DocumentSegment>> vectorResults = vectorStore.search(vectorReq);

        System.out.println("【向量搜索】返回 " + vectorResults.size() + " 条:");
        vectorResults.forEach(r -> {
            System.out.println("   - " + r.getDocument().getId() + " (score: " + String.format("%.4f", r.getScore()) + ")");
            System.out.println("     " + r.getDocument().getContent());
        });
        System.out.println();

        // 2. BM25 搜索
        SearchRequest<DocumentSegment> bm25Req = SearchRequest.<DocumentSegment>builder()
                .query(queryText)
                .searchType(SearchType.BM25)
                .topK(topK)
                .documentClass(DocumentSegment.class)
                .build();
        List<SearchResult<DocumentSegment>> bm25Results = vectorStore.search(bm25Req);

        System.out.println("【BM25 搜索】返回 " + bm25Results.size() + " 条:");
        bm25Results.forEach(r -> {
            System.out.println("   - " + r.getDocument().getId() + " (score: " + String.format("%.4f", r.getScore()) + ")");
            System.out.println("     " + r.getDocument().getContent());
        });
        System.out.println();

        // 3. 混合搜索
        SearchRequest<DocumentSegment> hybridReq = SearchRequest.<DocumentSegment>builder()
                .query(queryText)
                .searchType(SearchType.HYBRID)
                .vectorWeight(0.5f)
                .bm25Weight(0.5f)
                .topK(topK)
                .documentClass(DocumentSegment.class)
                .build();
        List<SearchResult<DocumentSegment>> hybridResults = vectorStore.search(hybridReq);

        System.out.println("【混合搜索】（各50%）返回 " + hybridResults.size() + " 条:");
        hybridResults.forEach(r -> {
            System.out.println("   - " + r.getDocument().getId() + " (score: " + String.format("%.4f", r.getScore()) + ")");
            System.out.println("     " + r.getDocument().getContent());
        });
    }

    // ==================== 4.10 批量搜索（一次 RPC，多 query）====================

    @Test
    @Order(39)
    @DisplayName("4.10 批量向量搜索（多个 query 文本，一次 RPC）")
    void testBatchVectorSearch() {
        String partition = DocumentSegment.getPartitionName(KNOWLEDGE_1);

        // 三个查询文本一次性发送给 Milvus，SDK 层面并行处理
        BatchSearchRequest<DocumentSegment> request = BatchSearchRequest.<DocumentSegment>builder()
                .queries(Arrays.asList("Java 编程语言", "人工智能技术", "Spring Boot 框架"))
                .topK(2)
                .inPartition(partition)
                .documentClass(DocumentSegment.class)
                .build();

        List<List<SearchResult<DocumentSegment>>> allResults = vectorStore.batchSearch(request);

        // 外层结果数 == 输入 query 数
        assertEquals(3, allResults.size(), "批量搜索应返回 3 组结果");

        List<String> queries = Arrays.asList("Java 编程语言", "人工智能技术", "Spring Boot 框架");
        for (int i = 0; i < allResults.size(); i++) {
            List<SearchResult<DocumentSegment>> group = allResults.get(i);
            assertFalse(group.isEmpty(), "每组结果不应为空: query=" + queries.get(i));
            System.out.println("✅ 批量查询[" + i + "] \"" + queries.get(i) + "\"，返回 " + group.size() + " 条:");
            group.forEach(r -> {
                DocumentSegment seg = r.getDocument();
                System.out.println("   - " + seg.getId()
                        + " (score: " + String.format("%.4f", r.getScore()) + ")");
                System.out.println("     content: " + seg.getContent());
            });
        }
    }

    @Test
    @Order(45)
    @DisplayName("4.11 批量 BM25 搜索（多个 query 文本，一次 RPC）")
    void testBatchBM25Search() {
        String partition = DocumentSegment.getPartitionName(KNOWLEDGE_1);

        BatchSearchRequest<DocumentSegment> request = BatchSearchRequest.<DocumentSegment>builder()
                .queries(Arrays.asList("框架", "深度学习"))
                .searchType(SearchType.BM25)
                .topK(2)
                .inPartition(partition)
                .documentClass(DocumentSegment.class)
                .build();

        List<List<SearchResult<DocumentSegment>>> allResults = vectorStore.batchSearch(request);

        assertEquals(2, allResults.size(), "批量 BM25 搜索应返回 2 组结果");

        List<String> queries = Arrays.asList("框架", "深度学习");
        for (int i = 0; i < allResults.size(); i++) {
            System.out.println("✅ BM25 批量查询[" + i + "] \"" + queries.get(i)
                    + "\"，返回 " + allResults.get(i).size() + " 条:");
            allResults.get(i).forEach(r ->
                    System.out.println("   - " + r.getDocument().getId()
                            + " (score: " + String.format("%.4f", r.getScore()) + ")"
                            + " | " + r.getDocument().getContent()));
        }
    }

    @Test
    @Order(46)
    @DisplayName("4.12 批量混合搜索（向量 + BM25，多个 query，一次 RPC）")
    void testBatchHybridSearch() {
        String partition = DocumentSegment.getPartitionName(KNOWLEDGE_1);

        BatchSearchRequest<DocumentSegment> request = BatchSearchRequest.<DocumentSegment>builder()
                .queries(Arrays.asList("人工智能 机器学习", "Java Spring"))
                .searchType(SearchType.HYBRID)
                .vectorWeight(0.6f)
                .bm25Weight(0.4f)
                .topK(2)
                .inPartition(partition)
                .documentClass(DocumentSegment.class)
                .build();

        List<List<SearchResult<DocumentSegment>>> allResults = vectorStore.batchSearch(request);

        assertEquals(2, allResults.size(), "批量混合搜索应返回 2 组结果");

        List<String> queries = Arrays.asList("人工智能 机器学习", "Java Spring");
        for (int i = 0; i < allResults.size(); i++) {
            System.out.println("✅ 混合批量查询[" + i + "] \"" + queries.get(i)
                    + "\"，返回 " + allResults.get(i).size() + " 条:");
            allResults.get(i).forEach(r ->
                    System.out.println("   - " + r.getDocument().getId()
                            + " (score: " + String.format("%.4f", r.getScore()) + ")"
                            + " | " + r.getDocument().getContent()));
        }
    }

    // ==================== 5. 分区数据管理 ====================

    @Test
    @Order(40)
    @DisplayName("5.1 Upsert（自动嵌入）")
    void testUpsertInPartition() throws InterruptedException {
        String partition = DocumentSegment.getPartitionName(KNOWLEDGE_1);

        DocumentSegment segment = DocumentSegment.builder()
                .id("upsert_test")
                .fileId(FILE_1)
                .content("这是通过 Upsert 插入的新内容，会自动进行向量化")
                .build();

        vectorStore.upsert(Collections.singletonList(segment), partition);

        // Milvus 写入 WAL 后，数据需要一小段时间才能在 growing segment 中可见
        Thread.sleep(2000);

        // 使用 query + filter 验证数据存在：
        // query 扫描 growing/sealed segment，比 getById(GetReq) 对新写入数据更即时可见
        QueryRequest<DocumentSegment> queryReq = QueryRequest.<DocumentSegment>builder()
                .filter("id == \"upsert_test\"")
                .partitionName(partition)
                .documentClass(DocumentSegment.class)
                .build();
        List<DocumentSegment> results = vectorStore.query(queryReq);
        assertFalse(results.isEmpty(), "Upsert 后应该能查到数据");

        System.out.println("✅ Upsert 成功（自动嵌入）");
    }

    @Test
    @Order(41)
    @DisplayName("5.2 删除数据")
    void testDeleteInPartition() throws InterruptedException {
        String partition = DocumentSegment.getPartitionName(KNOWLEDGE_1);

        vectorStore.delete(Collections.singletonList("upsert_test"), partition);

        // delete delta log 写入后需短暂等待才能被 query 看到
        Thread.sleep(2000);

        // 使用 query + filter 验证数据已删除：
        // query 会立即合并 delete delta log，无需 flush 即可看到删除结果
        QueryRequest<Document> queryReq = QueryRequest.<Document>builder()
                .filter("id == \"upsert_test\"")
                .partitionName(partition)
                .build();
        List<Document> results = vectorStore.query(queryReq);
        assertTrue(results.isEmpty(), "删除后 query 应该查不到数据");


        System.out.println("✅ 删除成功");
    }

    @Test
    @Order(42)
    @DisplayName("5.3 根据文档ID删除所有片段")
    void testDeleteByFileId() throws InterruptedException {
        String partition = DocumentSegment.getPartitionName(KNOWLEDGE_1);

        // 先插入测试数据
        String testFileId = "file_to_delete";
        List<DocumentSegment> segments = createTestSegments(testFileId, 0, 2);
        vectorStore.add(new ArrayList<>(segments), partition);

        // Milvus flush 限速 0.1/s（最小间隔 10s），先等待足够时间再 flush
        Thread.sleep(12000);

        // flush：将 growing segment 封闭，使新插入的数据对 count(*) 可见
        vectorStore.flush();

        // 等待 Milvus 完成索引构建
        Thread.sleep(2000);

        long countBefore = vectorStore.count(partition);
        System.out.println("   删除前: " + countBefore);

        // 删除
        String filter = DocumentSegment.filterByFileId(testFileId);
        vectorStore.deleteByFilter(filter, partition);

        // deleteByFilter 写入 delta log 后需短暂等待才能被 query 看到
        Thread.sleep(2000);

        // 使用 query + filter 验证删除结果：query 立即合并 delete delta，无需第二次 flush
        List<Document> remaining = vectorStore.query(filter, Document.class);
        assertTrue(remaining.isEmpty(), "deleteByFilter 后 query 应查不到 file_id=" + testFileId + " 的文档");

        System.out.println("   删除后: " + (countBefore - 2) + "（已通过 query 验证删除成功）");
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

