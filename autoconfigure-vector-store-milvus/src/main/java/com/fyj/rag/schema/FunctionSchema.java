package com.fyj.rag.schema;

import io.milvus.common.clientenum.FunctionType;
import io.milvus.v2.service.collection.request.CreateCollectionReq;
import lombok.Builder;
import lombok.Data;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Function Schema 定义
 * <p>
 * 用于定义 Collection 中的 Function，如 BM25 全文检索函数
 */
@Data
@Builder
public class FunctionSchema {

    /**
     * Function 名称
     */
    private String name;

    /**
     * Function 类型
     */
    private FunctionType type;

    /**
     * 输入字段名称列表
     */
    private List<String> inputFieldNames;

    /**
     * 输出字段名称列表
     */
    private List<String> outputFieldNames;

    /**
     * 额外参数
     */
    @Builder.Default
    private Map<String, Object> params = new HashMap<>();

    // ========== 静态工厂方法 ==========

    /**
     * 创建 BM25 Function
     * <p>
     * BM25 Function 用于将文本字段转换为稀疏向量，支持全文检索
     *
     * @param name            Function 名称
     * @param inputFieldName  输入字段名称（文本字段，如 content）
     * @param outputFieldName 输出字段名称（稀疏向量字段，如 sparse）
     * @return FunctionSchema
     */
    public static FunctionSchema bm25(String name, String inputFieldName, String outputFieldName) {
        return FunctionSchema.builder()
                .name(name)
                .type(FunctionType.BM25)
                .inputFieldNames(Collections.singletonList(inputFieldName))
                .outputFieldNames(Collections.singletonList(outputFieldName))
                .build();
    }

    /**
     * 创建 BM25 Function（使用默认名称）
     *
     * @param inputFieldName  输入字段名称（文本字段）
     * @param outputFieldName 输出字段名称（稀疏向量字段）
     * @return FunctionSchema
     */
    public static FunctionSchema bm25(String inputFieldName, String outputFieldName) {
        return bm25("bm25_" + inputFieldName + "_to_" + outputFieldName, inputFieldName, outputFieldName);
    }

    /**
     * 转换为 Milvus SDK 的 Function
     */
    public CreateCollectionReq.Function toFunction() {
        return CreateCollectionReq.Function.builder()
                .name(name)
                .functionType(type)
                .inputFieldNames(inputFieldNames)
                .outputFieldNames(outputFieldNames)
                .build();
    }
}

