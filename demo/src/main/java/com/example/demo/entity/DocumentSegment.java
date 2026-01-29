package com.example.demo.entity;

import com.fyj.rag.schema.CollectionSchema;
import com.fyj.rag.schema.FieldSchema;
import com.fyj.rag.schema.IndexSchema;
import com.fyj.rag.vectorstore.Document;
import com.google.gson.JsonObject;
import io.milvus.v2.common.IndexParam;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 文档片段实体
 * <p>
 * 继承自 Document，可直接用于 MilvusVectorStore 操作
 * 对应 Milvus Collection: document_segments
 */
@Data
@SuperBuilder
@NoArgsConstructor
@EqualsAndHashCode(callSuper = true)
public class DocumentSegment extends Document {

    /**
     * 所属知识库ID
     */
    private String knowledgeId;

    /**
     * 所属文档ID
     */
    private String fileId;

    // ==================== Collection 配置常量 ====================

    /**
     * Collection 名称
     */
    public static final String COLLECTION_NAME = "document_segments";

    /**
     * 字段名称常量
     */
    public static final String FIELD_ID = "id";
    public static final String FIELD_KNOWLEDGE_ID = "knowledge_id";
    public static final String FIELD_FILE_ID = "file_id";
    public static final String FIELD_CONTENT = "content";
    public static final String FIELD_EMBEDDING = "embedding";
    public static final String FIELD_METADATA = "metadata";

    /**
     * 默认向量维度（OpenAI text-embedding-ada-002 为 1536）
     */
    public static final int DEFAULT_DIMENSION = 1536;

    // ==================== 构造方法 ====================

    public DocumentSegment(String id, String knowledgeId, String fileId, String content,
                           List<Float> embedding, Map<String, Object> metadata) {
        super(id, content, embedding, metadata != null ? metadata : new HashMap<>());
        this.knowledgeId = knowledgeId;
        this.fileId = fileId;
    }

    // ==================== 覆盖 toJsonObject 方法 ====================

    /**
     * 转换为 JsonObject，添加 knowledgeId 和 fileId 字段
     */
    @Override
    public JsonObject toJsonObject(String idField, String contentField, String embeddingField, String metadataField) {
        JsonObject json = super.toJsonObject(idField, contentField, embeddingField, metadataField);

        // 添加额外字段
        if (this.knowledgeId != null) {
            json.addProperty(FIELD_KNOWLEDGE_ID, this.knowledgeId);
        }
        if (this.fileId != null) {
            json.addProperty(FIELD_FILE_ID, this.fileId);
        }

        return json;
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
                .field(FieldSchema.varchar(FIELD_KNOWLEDGE_ID, 64))          // 知识库ID
                .field(FieldSchema.varchar(FIELD_FILE_ID, 64))               // 文档ID
                .field(FieldSchema.varchar(FIELD_CONTENT, 65535))            // 内容，最大65535字符
                .field(FieldSchema.floatVector(FIELD_EMBEDDING, dimension))  // 向量
                .field(FieldSchema.json(FIELD_METADATA))                     // 元数据
                .enableDynamicField(false)
                .build();
    }

    /**
     * 创建默认索引（AUTOINDEX + COSINE）
     */
    public static IndexSchema createIndex() {
        return IndexSchema.autoIndex(FIELD_EMBEDDING, IndexParam.MetricType.COSINE);
    }

    /**
     * 创建 HNSW 索引（推荐用于高精度搜索）
     */
    public static IndexSchema createHnswIndex(int m, int efConstruction) {
        return IndexSchema.hnsw(FIELD_EMBEDDING, IndexParam.MetricType.COSINE, m, efConstruction);
    }

    /**
     * 创建 IVF_FLAT 索引（推荐用于大数据量）
     */
    public static IndexSchema createIvfFlatIndex(int nlist) {
        return IndexSchema.ivfFlat(FIELD_EMBEDDING, IndexParam.MetricType.COSINE, nlist);
    }

    // ==================== 静态工厂方法 ====================

    /**
     * 创建文档片段
     */
    public static DocumentSegment of(String id, String knowledgeId, String fileId,
                                     String content, List<Float> embedding) {
        return new DocumentSegment(id, knowledgeId, fileId, content, embedding, new HashMap<>());
    }

    /**
     * 创建文档片段（带元数据）
     */
    public static DocumentSegment of(String id, String knowledgeId, String fileId,
                                     String content, List<Float> embedding, Map<String, Object> metadata) {
        return new DocumentSegment(id, knowledgeId, fileId, content, embedding, metadata);
    }

    /**
     * 从 Document 转换（需要从 metadata 中提取 knowledgeId 和 fileId）
     */
    public static DocumentSegment fromDocument(Document document) {
        Map<String, Object> meta = document.getMetadata();
        String knowledgeId = null;
        String fileId = null;

        if (meta != null) {
            if (meta.containsKey(FIELD_KNOWLEDGE_ID)) {
                knowledgeId = String.valueOf(meta.get(FIELD_KNOWLEDGE_ID));
            }
            if (meta.containsKey(FIELD_FILE_ID)) {
                fileId = String.valueOf(meta.get(FIELD_FILE_ID));
            }
        }

        return new DocumentSegment(
                document.getId(),
                knowledgeId,
                fileId,
                document.getContent(),
                document.getEmbedding(),
                meta
        );
    }

    // ==================== 过滤表达式构建方法 ====================

    /**
     * 构建按知识库ID过滤的表达式
     */
    public static String filterByKnowledgeId(String knowledgeId) {
        return String.format("%s == \"%s\"", FIELD_KNOWLEDGE_ID, knowledgeId);
    }

    /**
     * 构建按文档ID过滤的表达式
     */
    public static String filterByFileId(String fileId) {
        return String.format("%s == \"%s\"", FIELD_FILE_ID, fileId);
    }

    /**
     * 构建按知识库ID和文档ID过滤的表达式
     */
    public static String filterByKnowledgeIdAndFileId(String knowledgeId, String fileId) {
        return String.format("%s == \"%s\" && %s == \"%s\"",
                FIELD_KNOWLEDGE_ID, knowledgeId, FIELD_FILE_ID, fileId);
    }

    /**
     * 构建按多个知识库ID过滤的表达式（用于跨知识库搜索）
     */
    public static String filterByKnowledgeIds(List<String> knowledgeIds) {
        return String.format("%s in [\"%s\"]",
                FIELD_KNOWLEDGE_ID,
                String.join("\", \"", knowledgeIds));
    }

    /**
     * 构建按多个文档ID过滤的表达式
     */
    public static String filterByFileIds(List<String> fileIds) {
        return String.format("%s in [\"%s\"]",
                FIELD_FILE_ID,
                String.join("\", \"", fileIds));
    }
}

