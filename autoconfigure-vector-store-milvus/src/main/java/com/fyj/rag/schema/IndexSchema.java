package com.fyj.rag.schema;

import io.milvus.v2.common.IndexParam;
import lombok.Builder;
import lombok.Data;

import java.util.HashMap;
import java.util.Map;

/**
 * 索引 Schema 定义
 */
@Data
@Builder
public class IndexSchema {

    /**
     * 索引名称
     */
    private String indexName;

    /**
     * 字段名称
     */
    private String fieldName;

    /**
     * 索引类型
     */
    private IndexParam.IndexType indexType;

    /**
     * 度量类型
     */
    private IndexParam.MetricType metricType;

    /**
     * 额外参数
     */
    @Builder.Default
    private Map<String, Object> extraParams = new HashMap<>();

    // ========== 静态工厂方法 ==========

    /**
     * 创建 AUTOINDEX 索引（自动选择最优索引类型）
     */
    public static IndexSchema autoIndex(String fieldName, IndexParam.MetricType metricType) {
        return IndexSchema.builder()
                .fieldName(fieldName)
                .indexType(IndexParam.IndexType.AUTOINDEX)
                .metricType(metricType)
                .build();
    }

    /**
     * 创建 IVF_FLAT 索引
     */
    public static IndexSchema ivfFlat(String fieldName, IndexParam.MetricType metricType, int nlist) {
        Map<String, Object> params = new HashMap<>();
        params.put("nlist", nlist);
        return IndexSchema.builder()
                .fieldName(fieldName)
                .indexType(IndexParam.IndexType.IVF_FLAT)
                .metricType(metricType)
                .extraParams(params)
                .build();
    }

    /**
     * 创建 IVF_SQ8 索引
     */
    public static IndexSchema ivfSq8(String fieldName, IndexParam.MetricType metricType, int nlist) {
        Map<String, Object> params = new HashMap<>();
        params.put("nlist", nlist);
        return IndexSchema.builder()
                .fieldName(fieldName)
                .indexType(IndexParam.IndexType.IVF_SQ8)
                .metricType(metricType)
                .extraParams(params)
                .build();
    }

    /**
     * 创建 IVF_PQ 索引
     */
    public static IndexSchema ivfPq(String fieldName, IndexParam.MetricType metricType, int nlist, int m, int nbits) {
        Map<String, Object> params = new HashMap<>();
        params.put("nlist", nlist);
        params.put("m", m);
        params.put("nbits", nbits);
        return IndexSchema.builder()
                .fieldName(fieldName)
                .indexType(IndexParam.IndexType.IVF_PQ)
                .metricType(metricType)
                .extraParams(params)
                .build();
    }

    /**
     * 创建 HNSW 索引
     */
    public static IndexSchema hnsw(String fieldName, IndexParam.MetricType metricType, int m, int efConstruction) {
        Map<String, Object> params = new HashMap<>();
        params.put("M", m);
        params.put("efConstruction", efConstruction);
        return IndexSchema.builder()
                .fieldName(fieldName)
                .indexType(IndexParam.IndexType.HNSW)
                .metricType(metricType)
                .extraParams(params)
                .build();
    }

    /**
     * 创建 FLAT 索引（暴力搜索，适合小数据量）
     */
    public static IndexSchema flat(String fieldName, IndexParam.MetricType metricType) {
        return IndexSchema.builder()
                .fieldName(fieldName)
                .indexType(IndexParam.IndexType.FLAT)
                .metricType(metricType)
                .build();
    }

    /**
     * 创建 DISKANN 索引（磁盘索引，适合大数据量）
     */
    public static IndexSchema diskAnn(String fieldName, IndexParam.MetricType metricType) {
        return IndexSchema.builder()
                .fieldName(fieldName)
                .indexType(IndexParam.IndexType.DISKANN)
                .metricType(metricType)
                .build();
    }

    /**
     * 创建 GPU_IVF_FLAT 索引
     */
    public static IndexSchema gpuIvfFlat(String fieldName, IndexParam.MetricType metricType, int nlist) {
        Map<String, Object> params = new HashMap<>();
        params.put("nlist", nlist);
        return IndexSchema.builder()
                .fieldName(fieldName)
                .indexType(IndexParam.IndexType.GPU_IVF_FLAT)
                .metricType(metricType)
                .extraParams(params)
                .build();
    }

    /**
     * 创建 GPU_IVF_PQ 索引
     */
    public static IndexSchema gpuIvfPq(String fieldName, IndexParam.MetricType metricType, int nlist, int m, int nbits) {
        Map<String, Object> params = new HashMap<>();
        params.put("nlist", nlist);
        params.put("m", m);
        params.put("nbits", nbits);
        return IndexSchema.builder()
                .fieldName(fieldName)
                .indexType(IndexParam.IndexType.GPU_IVF_PQ)
                .metricType(metricType)
                .extraParams(params)
                .build();
    }

    /**
     * 转换为 Milvus SDK 的 IndexParam
     */
    public IndexParam toIndexParam() {
        return IndexParam.builder()
                .indexName(indexName)
                .fieldName(fieldName)
                .indexType(indexType)
                .metricType(metricType)
                .extraParams(extraParams)
                .build();
    }
}

