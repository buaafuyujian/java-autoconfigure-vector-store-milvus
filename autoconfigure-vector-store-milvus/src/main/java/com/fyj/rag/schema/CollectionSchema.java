package com.fyj.rag.schema;

import lombok.Builder;
import lombok.Data;

import java.util.ArrayList;
import java.util.List;

/**
 * Collection Schema 定义
 */
@Data
@Builder
public class CollectionSchema {

    /**
     * Collection 描述
     */
    private String description;

    /**
     * 字段列表
     */
    @Builder.Default
    private List<FieldSchema> fields = new ArrayList<>();

    /**
     * 是否启用动态字段
     */
    @Builder.Default
    private boolean enableDynamicField = true;

    /**
     * Function 列表（如 BM25）
     */
    @Builder.Default
    private List<FunctionSchema> functions = new ArrayList<>();

    /**
     * 添加字段
     */
    public CollectionSchema addField(FieldSchema field) {
        this.fields.add(field);
        return this;
    }

    /**
     * 添加 Function
     */
    public CollectionSchema addFunction(FunctionSchema function) {
        this.functions.add(function);
        return this;
    }

    // ========== 静态工厂方法 ==========

    /**
     * 创建默认的向量存储 Schema
     * 包含: id(主键), content(文本), embedding(向量), metadata(JSON)
     */
    public static CollectionSchema defaultSchema(int dimension) {
        return CollectionSchema.builder()
                .description("Default vector store collection")
                .enableDynamicField(true)
                .fields(new ArrayList<>(List.of(
                        FieldSchema.primaryKey("id"),
                        FieldSchema.varchar("content", 65535),
                        FieldSchema.floatVector("embedding", dimension),
                        FieldSchema.json("metadata")
                )))
                .build();
    }

    /**
     * 创建简单的向量存储 Schema（仅包含 id 和 embedding）
     */
    public static CollectionSchema simpleSchema(int dimension) {
        return CollectionSchema.builder()
                .description("Simple vector store collection")
                .enableDynamicField(true)
                .fields(new ArrayList<>(List.of(
                        FieldSchema.primaryKey("id"),
                        FieldSchema.floatVector("embedding", dimension)
                )))
                .build();
    }

    /**
     * 创建自定义 Schema Builder
     */
    public static SchemaBuilder create() {
        return new SchemaBuilder();
    }

    /**
     * Schema 构建器
     */
    public static class SchemaBuilder {
        private String description;
        private final List<FieldSchema> fields = new ArrayList<>();
        private final List<FunctionSchema> functions = new ArrayList<>();
        private boolean enableDynamicField = true;

        public SchemaBuilder description(String description) {
            this.description = description;
            return this;
        }

        public SchemaBuilder field(FieldSchema field) {
            this.fields.add(field);
            return this;
        }

        public SchemaBuilder primaryKey(String name) {
            this.fields.add(FieldSchema.primaryKey(name));
            return this;
        }

        public SchemaBuilder primaryKeyInt64(String name, boolean autoId) {
            this.fields.add(FieldSchema.primaryKeyInt64(name, autoId));
            return this;
        }

        public SchemaBuilder varchar(String name, int maxLength) {
            this.fields.add(FieldSchema.varchar(name, maxLength));
            return this;
        }

        /**
         * 添加 VARCHAR 字段（启用分词器，用于 BM25 全文检索）
         */
        public SchemaBuilder varcharWithAnalyzer(String name, int maxLength) {
            this.fields.add(FieldSchema.varcharWithAnalyzer(name, maxLength));
            return this;
        }

        public SchemaBuilder floatVector(String name, int dimension) {
            this.fields.add(FieldSchema.floatVector(name, dimension));
            return this;
        }

        public SchemaBuilder sparseFloatVector(String name) {
            this.fields.add(FieldSchema.sparseFloatVector(name));
            return this;
        }

        public SchemaBuilder json(String name) {
            this.fields.add(FieldSchema.json(name));
            return this;
        }

        public SchemaBuilder int64(String name) {
            this.fields.add(FieldSchema.int64(name));
            return this;
        }

        public SchemaBuilder bool(String name) {
            this.fields.add(FieldSchema.bool(name));
            return this;
        }

        public SchemaBuilder enableDynamicField(boolean enable) {
            this.enableDynamicField = enable;
            return this;
        }

        /**
         * 添加 Function
         */
        public SchemaBuilder function(FunctionSchema function) {
            this.functions.add(function);
            return this;
        }

        /**
         * 添加 BM25 Function
         *
         * @param name           Function 名称
         * @param inputFieldName 输入字段名称（文本字段）
         * @param outputFieldName 输出字段名称（稀疏向量字段）
         */
        public SchemaBuilder bm25Function(String name, String inputFieldName, String outputFieldName) {
            this.functions.add(FunctionSchema.bm25(name, inputFieldName, outputFieldName));
            return this;
        }

        /**
         * 添加 BM25 Function（使用默认名称）
         *
         * @param inputFieldName 输入字段名称（文本字段）
         * @param outputFieldName 输出字段名称（稀疏向量字段）
         */
        public SchemaBuilder bm25Function(String inputFieldName, String outputFieldName) {
            this.functions.add(FunctionSchema.bm25(inputFieldName, outputFieldName));
            return this;
        }

        public CollectionSchema build() {
            return CollectionSchema.builder()
                    .description(description)
                    .fields(fields)
                    .functions(functions)
                    .enableDynamicField(enableDynamicField)
                    .build();
        }
    }
}

