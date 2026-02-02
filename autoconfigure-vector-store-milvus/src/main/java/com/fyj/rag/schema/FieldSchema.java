package com.fyj.rag.schema;

import io.milvus.v2.common.DataType;
import lombok.Builder;
import lombok.Data;

import java.util.Map;

/**
 * 字段 Schema 定义
 */
@Data
@Builder
public class FieldSchema {

    /**
     * 字段名称
     */
    private String name;

    /**
     * 数据类型
     */
    private DataType dataType;

    /**
     * 是否为主键
     */
    @Builder.Default
    private boolean isPrimaryKey = false;

    /**
     * 主键是否自动生成ID
     */
    @Builder.Default
    private boolean autoId = false;

    /**
     * VARCHAR 类型的最大长度
     */
    private Integer maxLength;

    /**
     * VECTOR 类型的维度
     */
    private Integer dimension;

    /**
     * 字段描述
     */
    private String description;

    /**
     * 元素类型（用于 Array 类型）
     */
    private DataType elementType;

    /**
     * 数组最大容量
     */
    private Integer maxCapacity;

    /**
     * 是否启用分词器（用于 BM25 全文检索）
     * <p>
     * 当字段作为 BM25 Function 的输入时，必须设置为 true
     */
    @Builder.Default
    private boolean enableAnalyzer = false;

    /**
     * 分词器参数
     * <p>
     * 可以指定分词器类型，如 {"type": "chinese"} 用于中文分词
     */
    private Map<String, Object> analyzerParams;

    // ========== 静态工厂方法 ==========

    /**
     * 创建 VARCHAR 类型主键字段（自动生成 ID）
     */
    public static FieldSchema primaryKey(String name) {
        return FieldSchema.builder()
                .name(name)
                .dataType(DataType.VarChar)
                .isPrimaryKey(true)
                .autoId(false)
                .maxLength(64)
                .build();
    }

    /**
     * 创建 Int64 类型主键字段
     */
    public static FieldSchema primaryKeyInt64(String name, boolean autoId) {
        return FieldSchema.builder()
                .name(name)
                .dataType(DataType.Int64)
                .isPrimaryKey(true)
                .autoId(autoId)
                .build();
    }

    /**
     * 创建 VARCHAR 类型主键字段
     */
    public static FieldSchema primaryKeyVarchar(String name, int maxLength) {
        return FieldSchema.builder()
                .name(name)
                .dataType(DataType.VarChar)
                .isPrimaryKey(true)
                .autoId(false)
                .maxLength(maxLength)
                .build();
    }

    /**
     * 创建 VARCHAR 字段
     */
    public static FieldSchema varchar(String name, int maxLength) {
        return FieldSchema.builder()
                .name(name)
                .dataType(DataType.VarChar)
                .maxLength(maxLength)
                .build();
    }

    /**
     * 创建 VARCHAR 字段（启用分词器，用于 BM25 全文检索）
     * <p>
     * 当字段作为 BM25 Function 的输入时，必须启用分词器
     *
     * @param name      字段名称
     * @param maxLength 最大长度
     * @return FieldSchema
     */
    public static FieldSchema varcharWithAnalyzer(String name, int maxLength) {
        return FieldSchema.builder()
                .name(name)
                .dataType(DataType.VarChar)
                .maxLength(maxLength)
                .enableAnalyzer(true)
                .build();
    }

    /**
     * 创建 VARCHAR 字段（启用分词器，指定分词器参数）
     *
     * @param name           字段名称
     * @param maxLength      最大长度
     * @param analyzerParams 分词器参数，如 Map.of("type", "chinese")
     * @return FieldSchema
     */
    public static FieldSchema varcharWithAnalyzer(String name, int maxLength, Map<String, Object> analyzerParams) {
        return FieldSchema.builder()
                .name(name)
                .dataType(DataType.VarChar)
                .maxLength(maxLength)
                .enableAnalyzer(true)
                .analyzerParams(analyzerParams)
                .build();
    }

    /**
     * 创建 FloatVector 字段
     */
    public static FieldSchema floatVector(String name, int dimension) {
        return FieldSchema.builder()
                .name(name)
                .dataType(DataType.FloatVector)
                .dimension(dimension)
                .build();
    }

    /**
     * 创建 JSON 字段
     */
    public static FieldSchema json(String name) {
        return FieldSchema.builder()
                .name(name)
                .dataType(DataType.JSON)
                .build();
    }

    /**
     * 创建 Int64 字段
     */
    public static FieldSchema int64(String name) {
        return FieldSchema.builder()
                .name(name)
                .dataType(DataType.Int64)
                .build();
    }

    /**
     * 创建 Int32 字段
     */
    public static FieldSchema int32(String name) {
        return FieldSchema.builder()
                .name(name)
                .dataType(DataType.Int32)
                .build();
    }

    /**
     * 创建 Int16 字段
     */
    public static FieldSchema int16(String name) {
        return FieldSchema.builder()
                .name(name)
                .dataType(DataType.Int16)
                .build();
    }

    /**
     * 创建 Int8 字段
     */
    public static FieldSchema int8(String name) {
        return FieldSchema.builder()
                .name(name)
                .dataType(DataType.Int8)
                .build();
    }

    /**
     * 创建 Float 字段
     */
    public static FieldSchema floatField(String name) {
        return FieldSchema.builder()
                .name(name)
                .dataType(DataType.Float)
                .build();
    }

    /**
     * 创建 Double 字段
     */
    public static FieldSchema doubleField(String name) {
        return FieldSchema.builder()
                .name(name)
                .dataType(DataType.Double)
                .build();
    }

    /**
     * 创建 Bool 字段
     */
    public static FieldSchema bool(String name) {
        return FieldSchema.builder()
                .name(name)
                .dataType(DataType.Bool)
                .build();
    }

    /**
     * 创建 Array 字段
     */
    public static FieldSchema array(String name, DataType elementType, int maxCapacity) {
        return FieldSchema.builder()
                .name(name)
                .dataType(DataType.Array)
                .elementType(elementType)
                .maxCapacity(maxCapacity)
                .build();
    }

    /**
     * 创建 BinaryVector 字段
     */
    public static FieldSchema binaryVector(String name, int dimension) {
        return FieldSchema.builder()
                .name(name)
                .dataType(DataType.BinaryVector)
                .dimension(dimension)
                .build();
    }

    /**
     * 创建 SparseFloatVector 字段
     */
    public static FieldSchema sparseFloatVector(String name) {
        return FieldSchema.builder()
                .name(name)
                .dataType(DataType.SparseFloatVector)
                .build();
    }
}

