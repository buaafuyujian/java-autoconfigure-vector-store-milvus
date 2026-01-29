package com.fyj.rag.properties;

import io.milvus.common.clientenum.ConsistencyLevelEnum;
import io.milvus.v2.common.IndexParam;
import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * Milvus 配置属性
 */
@Data
@ConfigurationProperties(prefix = "spring.ai.vectorstore.milvus")
public class MilvusProperties {

    /**
     * Milvus 服务 URI，例如: http://localhost:19530
     */
    private String uri = "http://localhost:19530";

    /**
     * 认证 Token（用于 Milvus 云服务或启用认证的实例）
     */
    private String token;

    /**
     * 用户名
     */
    private String username;

    /**
     * 密码
     */
    private String password;

    /**
     * 数据库名称
     */
    private String databaseName = "default";

    /**
     * 连接超时时间（毫秒）
     */
    private long connectTimeoutMs = 10000;

    /**
     * 是否启用安全连接（TLS）
     */
    private boolean secure = false;

    /**
     * 空闲超时时间（毫秒）
     */
    private long idleTimeoutMs = 86400000;

    /**
     * 默认 Collection 名称
     */
    private String collectionName = "vector_store";

    /**
     * 向量维度（默认 1536，适配 OpenAI embedding）
     */
    private int embeddingDimension = 1536;

    /**
     * 度量类型
     */
    private IndexParam.MetricType metricType = IndexParam.MetricType.COSINE;

    /**
     * 索引类型
     */
    private IndexParam.IndexType indexType = IndexParam.IndexType.AUTOINDEX;

    /**
     * 是否在启动时自动初始化默认 Collection
     */
    private boolean initializeSchema = false;

    /**
     * 一致性级别
     */
    private ConsistencyLevelEnum consistencyLevel = ConsistencyLevelEnum.BOUNDED;
}

