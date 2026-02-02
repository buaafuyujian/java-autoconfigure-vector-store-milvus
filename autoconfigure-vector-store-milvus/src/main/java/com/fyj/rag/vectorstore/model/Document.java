package com.fyj.rag.vectorstore.model;

import com.fyj.rag.schema.CollectionSchema;
import com.fyj.rag.schema.FieldSchema;
import com.fyj.rag.schema.IndexSchema;
import com.fyj.rag.vectorstore.annotation.ExcludeField;
import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.annotations.SerializedName;
import io.milvus.v2.common.IndexParam;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 文档实体
 */
@Data
@SuperBuilder
@NoArgsConstructor
@AllArgsConstructor
public class Document {

    private static final Gson GSON = new Gson();

    /**
     * 文档唯一标识
     */
    private String id;

    /**
     * 文档内容
     */
    private String content;

    /**
     * 向量 embedding
     * 查询时默认不返回，以节省带宽
     */
    @ExcludeField
    private List<Float> embedding;

    /**
     * 稀疏向量（用于 BM25 全文检索）
     * <p>
     * 由 Milvus 的 BM25 Function 自动生成，无需手动设置
     * 查询时默认不返回
     */
    @ExcludeField
    private Map<Long, Float> sparse;

    /**
     * 元数据
     */
    private Map<String, Object> metadata;

    /**
     * 字段名称常量
     */
    public static final String FIELD_ID = "id";
    public static final String FIELD_CONTENT = "content";
    public static final String FIELD_EMBEDDING = "embedding";
    public static final String FIELD_SPARSE = "sparse";
    public static final String FIELD_METADATA = "metadata";
    public static final Integer FIELD_ID_LENGTH=128;
    public static final Integer FIELD_CONTENT_LENGTH=65535;

    /**
     * 创建文档
     */
    public static Document of(String id, String content, List<Float> embedding) {
        return Document.builder()
                .id(id)
                .content(content)
                .embedding(embedding)
                .metadata(new HashMap<>())
                .build();
    }

    /**
     * 创建文档（带元数据）
     */
    public static Document of(String id, String content, List<Float> embedding, Map<String, Object> metadata) {
        return Document.builder()
                .id(id)
                .content(content)
                .embedding(embedding)
                .metadata(metadata)
                .build();
    }

    /**
     * 添加元数据
     */
    public Document addMetadata(String key, Object value) {
        if (this.metadata == null) {
            this.metadata = new HashMap<>();
        }
        this.metadata.put(key, value);
        return this;
    }

    /**
     * 获取元数据，确保不为 null
     */
    public Map<String, Object> getMetadata() {
        if (this.metadata == null) {
            this.metadata = new HashMap<>();
        }
        return this.metadata;
    }

    /**
     * 转换为 JsonObject，用于插入 Milvus
     * 使用 Gson 序列化，子类的字段会自动包含
     */
    public JsonObject toJsonObject() {
        return GSON.toJsonTree(this).getAsJsonObject();
    }

    /**
     * 创建 Collection Schema（自定义维度）
     */
    public static CollectionSchema createSchema(int dimension) {
        return CollectionSchema.create()
                               .description("Document segments for RAG knowledge base")
                               .field(FieldSchema.primaryKeyVarchar(FIELD_ID, FIELD_ID_LENGTH))          // 主键
                               .field(FieldSchema.varchar(FIELD_CONTENT, FIELD_CONTENT_LENGTH))            // 内容
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
                               .field(FieldSchema.primaryKeyVarchar(FIELD_ID, FIELD_ID_LENGTH))
                               .field(FieldSchema.varcharWithAnalyzer(FIELD_CONTENT, FIELD_CONTENT_LENGTH))  // 启用分词器
                               .field(FieldSchema.floatVector(FIELD_EMBEDDING, dimension))
                               .field(FieldSchema.sparseFloatVector(FIELD_SPARSE))
                               .field(FieldSchema.json(FIELD_METADATA))
                               .bm25Function(FIELD_CONTENT, FIELD_SPARSE)  // BM25 Function: content -> sparse
                               .enableDynamicField(false)
                               .build();
    }

    /**
     * 创建 Collection Schema（自定义维度）
     */
    public static CollectionSchema.SchemaBuilder createSchemaBuilder(int dimension) {
        return CollectionSchema.create()
                               .description("Document segments for RAG knowledge base")
                               .field(FieldSchema.primaryKeyVarchar(FIELD_ID, FIELD_ID_LENGTH))          // 主键
                               .field(FieldSchema.varchar(FIELD_CONTENT, FIELD_CONTENT_LENGTH))            // 内容
                               .field(FieldSchema.floatVector(FIELD_EMBEDDING, dimension))  // 向量
                               .field(FieldSchema.sparseFloatVector(FIELD_SPARSE))          // 稀疏向量
                               .field(FieldSchema.json(FIELD_METADATA))                     // 元数据
                               .enableDynamicField(false);
    }

    /**
     * 创建支持 BM25 的 Schema Builder
     *
     * @param dimension 向量维度
     * @return SchemaBuilder
     */
    public static CollectionSchema.SchemaBuilder createSchemaBuilderWithBM25(int dimension) {
        return CollectionSchema.create()
                               .description("Document segments for RAG knowledge base with BM25 support")
                               .field(FieldSchema.primaryKeyVarchar(FIELD_ID, FIELD_ID_LENGTH))
                               .field(FieldSchema.varcharWithAnalyzer(FIELD_CONTENT, FIELD_CONTENT_LENGTH))  // 启用分词器
                               .field(FieldSchema.floatVector(FIELD_EMBEDDING, dimension))
                               .field(FieldSchema.sparseFloatVector(FIELD_SPARSE))
                               .field(FieldSchema.json(FIELD_METADATA))
                               .bm25Function(FIELD_CONTENT, FIELD_SPARSE)
                               .enableDynamicField(false);
    }

    /**
     * 创建默认索引（AUTOINDEX + COSINE）
     */
    public static IndexSchema createIndex() {
        return IndexSchema.autoIndex(FIELD_EMBEDDING, IndexParam.MetricType.COSINE);
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

    /**
     * 获取指定类需要输出的字段列表（排除带 @ExcludeField 注解的字段）
     * 会递归获取父类的字段
     * <p>
     * 注意：如果字段上有 @SerializedName 注解，则使用注解中的名称
     *
     * @param clazz 文档类型
     * @return 字段名列表（Milvus 中的实际字段名）
     */
    public static List<String> getOutputFields(Class<? extends Document> clazz) {
        List<String> fields = new ArrayList<>();
        Class<?> currentClass = clazz;

        while (currentClass != null && currentClass != Object.class) {
            for (Field field : currentClass.getDeclaredFields()) {
                // 排除静态字段、带 @ExcludeField 注解的字段
                if (Modifier.isStatic(field.getModifiers())) {
                    continue;
                }
                if (field.isAnnotationPresent(ExcludeField.class)) {
                    continue;
                }

                // 获取字段名：优先使用 @SerializedName 注解的值
                String fieldName = getSerializedFieldName(field);

                // 避免重复添加（子类可能覆盖父类字段）
                if (!fields.contains(fieldName)) {
                    fields.add(fieldName);
                }
            }
            currentClass = currentClass.getSuperclass();
        }

        return fields;
    }

    /**
     * 获取字段的序列化名称
     * 如果有 @SerializedName 注解则使用注解值，否则使用字段名
     *
     * @param field 字段
     * @return 序列化名称
     */
    private static String getSerializedFieldName(Field field) {
        SerializedName serializedName = field.getAnnotation(SerializedName.class);
        if (serializedName != null && !serializedName.value().isEmpty()) {
            return serializedName.value();
        }
        return field.getName();
    }
}

