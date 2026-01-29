package com.example.demo.entity;

import com.fyj.rag.schema.CollectionSchema;
import com.fyj.rag.schema.FieldSchema;
import com.fyj.rag.schema.IndexSchema;
import com.fyj.rag.vectorstore.Document;
import io.milvus.v2.common.IndexParam;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 文档片段实体
 * <p>
 * 对应 Milvus Collection: document_segments
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class DocumentSegment {

    /**
     * 片段唯一标识（主键）
     */
    private String id;

    /**
     * 所属知识库ID
     */
    private String knowledgeId;

    /**
     * 所属文档ID
     */
    private String fileId;

    /**
     * 片段原始内容
     */
    private String content;

    /**
     * 向量化后的内容
     */
    private List<Float> embedding;

    /**
     * 元数据（可存储：chunk_index, total_chunks, create_time 等）
     */
    @Builder.Default
    private Map<String, Object> metadata = new HashMap<>();

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

    // ==================== 转换方法 ====================

    /**
     * 转换为通用 Document 对象（用于 MilvusVectorStore 操作）
     */
    public Document toDocument() {
        Map<String, Object> meta = new HashMap<>(metadata != null ? metadata : new HashMap<>());
        meta.put(FIELD_KNOWLEDGE_ID, knowledgeId);
        meta.put(FIELD_FILE_ID, fileId);

        return Document.builder()
                .id(id)
                .content(content)
                .embedding(embedding)
                .metadata(meta)
                .build();
    }

    /**
     * 从通用 Document 对象转换
     */
    public static DocumentSegment fromDocument(Document document) {
        Map<String, Object> meta = document.getMetadata();

        DocumentSegmentBuilder builder = DocumentSegment.builder()
                .id(document.getId())
                .content(document.getContent())
                .embedding(document.getEmbedding());

        if (meta != null) {
            if (meta.containsKey(FIELD_KNOWLEDGE_ID)) {
                builder.knowledgeId(String.valueOf(meta.get(FIELD_KNOWLEDGE_ID)));
            }
            if (meta.containsKey(FIELD_FILE_ID)) {
                builder.fileId(String.valueOf(meta.get(FIELD_FILE_ID)));
            }

            // 移除已提取的字段，剩余作为 metadata
            Map<String, Object> remaining = new HashMap<>(meta);
            remaining.remove(FIELD_KNOWLEDGE_ID);
            remaining.remove(FIELD_FILE_ID);
            builder.metadata(remaining);
        }

        return builder.build();
    }

    // ==================== 辅助方法 ====================

    /**
     * 添加元数据
     */
    public DocumentSegment addMetadata(String key, Object value) {
        if (this.metadata == null) {
            this.metadata = new HashMap<>();
        }
        this.metadata.put(key, value);
        return this;
    }

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
}

