package com.example.demo;

import com.fyj.rag.client.MilvusClient;
import com.fyj.rag.schema.CollectionSchema;
import com.fyj.rag.schema.FieldSchema;
import com.fyj.rag.schema.IndexSchema;
import io.milvus.v2.common.DataType;
import io.milvus.v2.common.IndexParam;
import org.junit.jupiter.api.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Schema 创建功能测试
 *
 * 测试各种自定义 Schema 的创建方式
 */
@SpringBootTest
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class SchemaCreationTests {

    @Autowired
    private MilvusClient milvusClient;

    // ==================== 快速创建 Collection ====================

    @Test
    @Order(1)
    @DisplayName("测试快速创建 Collection（仅指定维度）")
    void testQuickCreateCollection() {
        String collectionName = "quick_create_test";

        // 清理
        if (milvusClient.hasCollection(collectionName)) {
            milvusClient.dropCollection(collectionName);
        }

        // 快速创建（自动创建 id 和 vector 字段）
        milvusClient.createCollection(collectionName, 128);

        assertTrue(milvusClient.hasCollection(collectionName));
        System.out.println("✅ Quick create collection: " + collectionName);

        // 清理
        milvusClient.dropCollection(collectionName);
    }

    @Test
    @Order(2)
    @DisplayName("测试快速创建 Collection（指定维度和度量类型）")
    void testQuickCreateCollectionWithMetric() {
        String collectionName = "quick_create_metric_test";

        if (milvusClient.hasCollection(collectionName)) {
            milvusClient.dropCollection(collectionName);
        }

        // 快速创建（指定度量类型）
        milvusClient.createCollection(collectionName, 128, IndexParam.MetricType.L2);

        assertTrue(milvusClient.hasCollection(collectionName));
        System.out.println("✅ Quick create collection with L2 metric: " + collectionName);

        milvusClient.dropCollection(collectionName);
    }

    // ==================== 使用默认 Schema ====================

    @Test
    @Order(10)
    @DisplayName("测试使用默认 Schema 创建 Collection")
    void testCreateWithDefaultSchema() {
        String collectionName = "default_schema_test";

        if (milvusClient.hasCollection(collectionName)) {
            milvusClient.dropCollection(collectionName);
        }

        // 使用默认 Schema（id, content, embedding, metadata）
        CollectionSchema schema = CollectionSchema.defaultSchema(256);
        IndexSchema index = IndexSchema.autoIndex("embedding", IndexParam.MetricType.COSINE);

        milvusClient.createCollection(collectionName, schema, index);

        assertTrue(milvusClient.hasCollection(collectionName));

        var desc = milvusClient.describeCollection(collectionName);
        System.out.println("✅ Created with default schema: " + collectionName);
        System.out.println("   Fields: " + desc.getFieldNames());

        milvusClient.dropCollection(collectionName);
    }

    @Test
    @Order(11)
    @DisplayName("测试使用简单 Schema 创建 Collection")
    void testCreateWithSimpleSchema() {
        String collectionName = "simple_schema_test";

        if (milvusClient.hasCollection(collectionName)) {
            milvusClient.dropCollection(collectionName);
        }

        // 使用简单 Schema（仅 id 和 embedding）
        CollectionSchema schema = CollectionSchema.simpleSchema(256);
        IndexSchema index = IndexSchema.autoIndex("embedding", IndexParam.MetricType.IP);

        milvusClient.createCollection(collectionName, schema, index);

        assertTrue(milvusClient.hasCollection(collectionName));

        var desc = milvusClient.describeCollection(collectionName);
        System.out.println("✅ Created with simple schema: " + collectionName);
        System.out.println("   Fields: " + desc.getFieldNames());

        milvusClient.dropCollection(collectionName);
    }

    // ==================== 自定义 Schema ====================

    @Test
    @Order(20)
    @DisplayName("测试使用 Builder 创建自定义 Schema")
    void testCreateWithCustomSchemaBuilder() {
        String collectionName = "custom_schema_builder_test";

        if (milvusClient.hasCollection(collectionName)) {
            milvusClient.dropCollection(collectionName);
        }

        // 使用 Builder 创建自定义 Schema
        CollectionSchema schema = CollectionSchema.create()
                .description("Custom collection with builder")
                .primaryKey("doc_id")
                .varchar("title", 512)
                .varchar("content", 65535)
                .floatVector("embedding", 384)
                .json("metadata")
                .int64("view_count")
                .bool("is_published")
                .enableDynamicField(true)
                .build();

        IndexSchema index = IndexSchema.hnsw("embedding", IndexParam.MetricType.COSINE, 16, 256);

        milvusClient.createCollection(collectionName, schema, index);

        assertTrue(milvusClient.hasCollection(collectionName));

        var desc = milvusClient.describeCollection(collectionName);
        System.out.println("✅ Created with custom schema builder: " + collectionName);
        System.out.println("   Fields: " + desc.getFieldNames());

        milvusClient.dropCollection(collectionName);
    }

    @Test
    @Order(21)
    @DisplayName("测试使用 FieldSchema 静态方法创建自定义 Schema")
    void testCreateWithFieldSchemaStaticMethods() {
        String collectionName = "custom_field_schema_test";

        if (milvusClient.hasCollection(collectionName)) {
            milvusClient.dropCollection(collectionName);
        }

        // 使用 FieldSchema 静态方法
        CollectionSchema schema = CollectionSchema.builder()
                .description("Custom collection with FieldSchema")
                .enableDynamicField(true)
                .build()
                .addField(FieldSchema.primaryKeyInt64("id", true))
                .addField(FieldSchema.varchar("text", 10000))
                .addField(FieldSchema.floatVector("vector", 512))
                .addField(FieldSchema.json("attrs"))
                .addField(FieldSchema.int32("score"))
                .addField(FieldSchema.doubleField("price"));

        IndexSchema index = IndexSchema.ivfFlat("vector", IndexParam.MetricType.L2, 128);

        milvusClient.createCollection(collectionName, schema, index);

        assertTrue(milvusClient.hasCollection(collectionName));

        var desc = milvusClient.describeCollection(collectionName);
        System.out.println("✅ Created with FieldSchema static methods: " + collectionName);
        System.out.println("   Fields: " + desc.getFieldNames());

        milvusClient.dropCollection(collectionName);
    }

    @Test
    @Order(22)
    @DisplayName("测试创建包含数组字段的 Schema")
    void testCreateWithArrayField() {
        String collectionName = "array_field_test";

        if (milvusClient.hasCollection(collectionName)) {
            milvusClient.dropCollection(collectionName);
        }

        CollectionSchema schema = CollectionSchema.builder()
                .description("Collection with array field")
                .enableDynamicField(false)
                .build()
                .addField(FieldSchema.primaryKey("id"))
                .addField(FieldSchema.floatVector("embedding", 128))
                .addField(FieldSchema.array("tags", DataType.VarChar, 100))
                .addField(FieldSchema.array("scores", DataType.Int32, 50));

        IndexSchema index = IndexSchema.autoIndex("embedding", IndexParam.MetricType.COSINE);

        milvusClient.createCollection(collectionName, schema, index);

        assertTrue(milvusClient.hasCollection(collectionName));

        var desc = milvusClient.describeCollection(collectionName);
        System.out.println("✅ Created with array fields: " + collectionName);
        System.out.println("   Fields: " + desc.getFieldNames());

        milvusClient.dropCollection(collectionName);
    }

    // ==================== 索引类型测试 ====================

    @Test
    @Order(30)
    @DisplayName("测试各种索引类型创建")
    void testDifferentIndexTypes() {
        String collectionName = "index_type_test";

        // AUTOINDEX
        testIndexType(collectionName + "_auto",
                IndexSchema.autoIndex("embedding", IndexParam.MetricType.COSINE));

        // IVF_FLAT
        testIndexType(collectionName + "_ivf_flat",
                IndexSchema.ivfFlat("embedding", IndexParam.MetricType.L2, 128));

        // HNSW
        testIndexType(collectionName + "_hnsw",
                IndexSchema.hnsw("embedding", IndexParam.MetricType.IP, 16, 256));

        // FLAT
        testIndexType(collectionName + "_flat",
                IndexSchema.flat("embedding", IndexParam.MetricType.COSINE));

        System.out.println("✅ All index types tested successfully");
    }

    private void testIndexType(String collectionName, IndexSchema indexSchema) {
        if (milvusClient.hasCollection(collectionName)) {
            milvusClient.dropCollection(collectionName);
        }

        CollectionSchema schema = CollectionSchema.simpleSchema(128);
        milvusClient.createCollection(collectionName, schema, indexSchema);

        assertTrue(milvusClient.hasCollection(collectionName));
        System.out.println("   - Created " + collectionName + " with " + indexSchema.getIndexType());

        milvusClient.dropCollection(collectionName);
    }

    // ==================== 二进制向量和稀疏向量测试 ====================

    @Test
    @Order(40)
    @DisplayName("测试创建二进制向量 Collection")
    void testCreateWithBinaryVector() {
        String collectionName = "binary_vector_test";

        if (milvusClient.hasCollection(collectionName)) {
            milvusClient.dropCollection(collectionName);
        }

        CollectionSchema schema = CollectionSchema.builder()
                .description("Collection with binary vector")
                .enableDynamicField(true)
                .build()
                .addField(FieldSchema.primaryKey("id"))
                .addField(FieldSchema.binaryVector("binary_embedding", 128));

        // 二进制向量使用 HAMMING 或 JACCARD 度量
        IndexSchema index = IndexSchema.builder()
                .fieldName("binary_embedding")
                .indexType(IndexParam.IndexType.BIN_FLAT)
                .metricType(IndexParam.MetricType.HAMMING)
                .build();

        milvusClient.createCollection(collectionName, schema, index);

        assertTrue(milvusClient.hasCollection(collectionName));
        System.out.println("✅ Created with binary vector: " + collectionName);

        milvusClient.dropCollection(collectionName);
    }

    @Test
    @Order(41)
    @DisplayName("测试创建稀疏向量 Collection")
    void testCreateWithSparseVector() {
        String collectionName = "sparse_vector_test";

        if (milvusClient.hasCollection(collectionName)) {
            milvusClient.dropCollection(collectionName);
        }

        CollectionSchema schema = CollectionSchema.builder()
                .description("Collection with sparse vector")
                .enableDynamicField(true)
                .build()
                .addField(FieldSchema.primaryKey("id"))
                .addField(FieldSchema.sparseFloatVector("sparse_embedding"));

        IndexSchema index = IndexSchema.builder()
                .fieldName("sparse_embedding")
                .indexType(IndexParam.IndexType.SPARSE_INVERTED_INDEX)
                .metricType(IndexParam.MetricType.IP)
                .build();

        milvusClient.createCollection(collectionName, schema, index);

        assertTrue(milvusClient.hasCollection(collectionName));
        System.out.println("✅ Created with sparse vector: " + collectionName);

        milvusClient.dropCollection(collectionName);
    }
}

