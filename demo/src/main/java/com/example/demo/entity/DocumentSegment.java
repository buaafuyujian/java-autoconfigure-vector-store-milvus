package com.example.demo.entity;

import com.fyj.rag.schema.CollectionSchema;
import com.fyj.rag.schema.FieldSchema;
import com.fyj.rag.schema.IndexSchema;
import com.fyj.rag.vectorstore.model.Document;
import com.google.gson.annotations.SerializedName;
import io.milvus.v2.common.IndexParam;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 文档片段实体
 * <p>
 * 继承自 Document，可直接用于 MilvusVectorStore 操作
 * 对应 Milvus Collection: document_segments
 * <p>
 * 注意：知识库通过分区（Partition）区分，不需要单独存储 knowledgeId
 */
@Data
@SuperBuilder
@NoArgsConstructor
@EqualsAndHashCode(callSuper = true)
public class DocumentSegment extends Document {

    /**
     * 所属文档ID
     */
    @SerializedName("file_id")
    private String fileId;

    // ==================== Collection 配置常量 ====================

    /**
     * Collection 名称
     */
    public static final String COLLECTION_NAME = "document_segments";

    /**
     * 字段名称常量
     */
    public static final String FIELD_FILE_ID = "file_id";

    /**
     * 默认向量维度（OpenAI text-embedding-ada-002 为 1536）
     */
    public static final int DEFAULT_DIMENSION = 1536;

    // ==================== 构造方法 ====================

    public DocumentSegment(String id, String fileId, String content,
                           List<Float> embedding, Map<String, Object> metadata) {
        super(id, content, embedding, null, metadata != null ? metadata : new HashMap<>());
        this.fileId = fileId;
    }

    public DocumentSegment(String id, String fileId, String content,
                           List<Float> embedding, Map<Long, Float> sparse, Map<String, Object> metadata) {
        super(id, content, embedding, sparse, metadata != null ? metadata : new HashMap<>());
        this.fileId = fileId;
    }


    // ==================== Schema 创建方法 ====================

    /**
     * 创建 Collection Schema（使用默认维度 1536）
     */
    public static CollectionSchema createSchema() {
        return createSchema(DEFAULT_DIMENSION);
    }

    /**
     * 创建 Collection Schema（自定义维度）
     */
    public static CollectionSchema createSchema(int dimension) {
        return CollectionSchema.create()
                .description("Document segments for RAG knowledge base")
                .field(FieldSchema.primaryKeyVarchar(FIELD_ID, 64))          // 主键，最大64字符
                .field(FieldSchema.varchar(FIELD_FILE_ID, 64))               // 文档ID
                .field(FieldSchema.varchar(FIELD_CONTENT, 65535))            // 内容，最大65535字符
                .field(FieldSchema.floatVector(FIELD_EMBEDDING, dimension))  // 向量
                .field(FieldSchema.sparseFloatVector(FIELD_SPARSE))          // 稀疏向量
                .field(FieldSchema.json(FIELD_METADATA))                     // 元数据
                .enableDynamicField(false)
                .build();
    }

    /**
     * 创建支持 BM25 的 Collection Schema（带 BM25 Function）
     * <p>
     * 此 Schema 会自动将 content 字段转换为稀疏向量存储到 sparse 字段
     * <p>
     * 注意：content 字段使用 varcharWithAnalyzer 启用分词器，这是 BM25 Function 的要求
     *
     * @param dimension 向量维度
     * @return CollectionSchema
     */
    public static CollectionSchema createSchemaWithBM25(int dimension) {
        return CollectionSchema.create()
                .description("Document segments for RAG knowledge base with BM25 support")
                .field(FieldSchema.primaryKeyVarchar(FIELD_ID, 64))
                .field(FieldSchema.varchar(FIELD_FILE_ID, 64))
                .field(FieldSchema.varcharWithAnalyzer(FIELD_CONTENT, 65535))  // 启用分词器
                .field(FieldSchema.floatVector(FIELD_EMBEDDING, dimension))
                .field(FieldSchema.sparseFloatVector(FIELD_SPARSE))
                .field(FieldSchema.json(FIELD_METADATA))
                .bm25Function(FIELD_CONTENT, FIELD_SPARSE)  // BM25 Function: content -> sparse
                .enableDynamicField(false)
                .build();
    }

    /**
     * 创建支持 BM25 的 Collection Schema（使用默认维度）
     */
    public static CollectionSchema createSchemaWithBM25() {
        return createSchemaWithBM25(DEFAULT_DIMENSION);
    }

    /**
     * 创建默认索引（AUTOINDEX + COSINE）
     */
    public static IndexSchema createIndex() {
        return IndexSchema.autoIndex(FIELD_EMBEDDING);
    }

    /**
     * 创建稀疏向量索引（用于 BM25 搜索）
     */
    public static IndexSchema createSparseIndex() {
        return IndexSchema.sparseInvertedIndex(FIELD_SPARSE);
    }

    /**
     * 创建所有索引（向量索引 + 稀疏向量索引）
     * <p>
     * 用于支持向量搜索、BM25搜索和混合搜索
     *
     * @return 索引列表
     */
    public static List<IndexSchema> createAllIndexes() {
        List<IndexSchema> indexes = new ArrayList<>();
        indexes.add(createIndex());       // 向量索引
        indexes.add(createSparseIndex()); // 稀疏向量索引
        return indexes;
    }

    /**
     * 创建 HNSW 索引（推荐用于高精度搜索）
     */
    public static IndexSchema createHnswIndex(int m, int efConstruction) {
        return IndexSchema.hnsw(FIELD_EMBEDDING, IndexParam.MetricType.COSINE, m, efConstruction);
    }

    /**
     * 创建 HNSW 索引 + 稀疏向量索引
     *
     * @param m              HNSW 参数
     * @param efConstruction HNSW 参数
     * @return 索引列表
     */
    public static List<IndexSchema> createHnswWithSparseIndexes(int m, int efConstruction) {
        List<IndexSchema> indexes = new ArrayList<>();
        indexes.add(createHnswIndex(m, efConstruction));
        indexes.add(createSparseIndex());
        return indexes;
    }

    /**
     * 创建 IVF_FLAT 索引（推荐用于大数据量）
     */
    public static IndexSchema createIvfFlatIndex(int nlist) {
        return IndexSchema.ivfFlat(FIELD_EMBEDDING, IndexParam.MetricType.COSINE, nlist);
    }

    /**
     * 创建 IVF_FLAT 索引 + 稀疏向量索引
     *
     * @param nlist IVF 参数
     * @return 索引列表
     */
    public static List<IndexSchema> createIvfFlatWithSparseIndexes(int nlist) {
        List<IndexSchema> indexes = new ArrayList<>();
        indexes.add(createIvfFlatIndex(nlist));
        indexes.add(createSparseIndex());
        return indexes;
    }

    // ==================== 静态工厂方法 ====================

    /**
     * 创建文档片段
     */
    public static DocumentSegment of(String id, String fileId, String content, List<Float> embedding) {
        return new DocumentSegment(id, fileId, content, embedding, new HashMap<>());
    }

    /**
     * 创建文档片段（带元数据）
     */
    public static DocumentSegment of(String id, String fileId, String content,
                                     List<Float> embedding, Map<String, Object> metadata) {
        return new DocumentSegment(id, fileId, content, embedding, metadata);
    }

    /**
     * 从 Document 转换（从 metadata 中提取 fileId）
     */
    public static DocumentSegment fromDocument(Document document) {
        Map<String, Object> meta = document.getMetadata();
        String fileId = null;

        if (meta != null && meta.containsKey(FIELD_FILE_ID)) {
            fileId = String.valueOf(meta.get(FIELD_FILE_ID));
        }

        return new DocumentSegment(
                document.getId(),
                fileId,
                document.getContent(),
                document.getEmbedding(),
                document.getSparse(),
                meta
        );
    }

    // ==================== 过滤表达式构建方法 ====================

    /**
     * 构建按文档ID过滤的表达式
     */
    public static String filterByFileId(String fileId) {
        return String.format("%s == \"%s\"", FIELD_FILE_ID, fileId);
    }

    /**
     * 构建按多个文档ID过滤的表达式
     */
    public static String filterByFileIds(List<String> fileIds) {
        return String.format("%s in [\"%s\"]",
                FIELD_FILE_ID,
                String.join("\", \"", fileIds));
    }

    // ==================== 分区相关方法 ====================

    /**
     * 根据知识库ID生成分区名称
     * <p>
     * 分区名称规则：partition_{knowledgeId}
     * 注意：分区名称只能包含字母、数字和下划线
     */
    public static String getPartitionName(String knowledgeId) {
        String safeName = knowledgeId.replaceAll("[^a-zA-Z0-9]", "_");
        return "partition_" + safeName;
    }

    /**
     * 从分区名称解析出知识库ID
     */
    public static String getKnowledgeIdFromPartition(String partitionName) {
        if (partitionName != null && partitionName.startsWith("partition_")) {
            return partitionName.substring("partition_".length());
        }
        return partitionName;
    }

    /**
     * 根据多个知识库ID生成分区名称列表
     */
    public static List<String> getPartitionNames(List<String> knowledgeIds) {
        return knowledgeIds.stream()
                .map(DocumentSegment::getPartitionName)
                .toList();
    }
}

