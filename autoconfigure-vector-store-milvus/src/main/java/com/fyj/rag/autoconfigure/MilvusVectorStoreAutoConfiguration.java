package com.fyj.rag.autoconfigure;

import com.fyj.rag.client.MilvusClient;
import com.fyj.rag.exception.ErrorCode;
import com.fyj.rag.exception.MilvusConnectionException;
import com.fyj.rag.properties.MilvusProperties;
import com.fyj.rag.schema.CollectionSchema;
import com.fyj.rag.schema.IndexSchema;
import com.fyj.rag.vectorstore.DefaultMilvusVectorStore;
import com.fyj.rag.vectorstore.MilvusVectorStore;
import io.milvus.v2.client.ConnectConfig;
import io.milvus.v2.client.MilvusClientV2;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;

/**
 * Milvus VectorStore 自动配置类
 */
@Slf4j
@AutoConfiguration
@ConditionalOnClass(MilvusClientV2.class)
@EnableConfigurationProperties(MilvusProperties.class)
public class MilvusVectorStoreAutoConfiguration {

    /**
     * 创建原始 MilvusClientV2 实例
     */
    @Bean
    @ConditionalOnMissingBean
    public MilvusClientV2 milvusClientV2(MilvusProperties properties) {
        try {
            ConnectConfig.ConnectConfigBuilder<?, ?> builder = ConnectConfig.builder()
                    .uri(properties.getUri())
                    .dbName(properties.getDatabaseName())
                    .connectTimeoutMs(properties.getConnectTimeoutMs())
                    .idleTimeoutMs(properties.getIdleTimeoutMs())
                    .secure(properties.isSecure());

            // Token 认证
            if (properties.getToken() != null && !properties.getToken().isEmpty()) {
                builder.token(properties.getToken());
            }

            // 用户名密码认证
            if (properties.getUsername() != null && !properties.getUsername().isEmpty()) {
                builder.username(properties.getUsername());
                builder.password(properties.getPassword());
            }

            MilvusClientV2 client = new MilvusClientV2(builder.build());
            log.info("Connected to Milvus server: {}, database: {}",
                    properties.getUri(), properties.getDatabaseName());
            return client;
        } catch (Exception e) {
            throw new MilvusConnectionException(ErrorCode.CONNECTION_FAILED,
                    "Failed to connect to Milvus server: " + properties.getUri(), e);
        }
    }

    /**
     * 创建 MilvusClient 包装实例
     */
    @Bean
    @ConditionalOnMissingBean
    public MilvusClient milvusClient(MilvusClientV2 milvusClientV2) {
        return new MilvusClient(milvusClientV2);
    }

    /**
     * 创建默认的 MilvusVectorStore 实例
     */
    @Bean
    @ConditionalOnMissingBean
    @ConditionalOnProperty(prefix = "spring.ai.vectorstore.milvus", name = "collection-name")
    public MilvusVectorStore milvusVectorStore(MilvusClient milvusClient, MilvusProperties properties) {
        String collectionName = properties.getCollectionName();

        // 如果启用了自动初始化 Schema
        if (properties.isInitializeSchema()) {
            initializeDefaultCollection(milvusClient, properties);
        }

        log.info("Created MilvusVectorStore for collection: {}", collectionName);
        return milvusClient.getVectorStore(collectionName);
    }

    /**
     * 初始化默认 Collection
     */
    private void initializeDefaultCollection(MilvusClient milvusClient, MilvusProperties properties) {
        String collectionName = properties.getCollectionName();

        if (!milvusClient.hasCollection(collectionName)) {
            log.info("Initializing default collection: {}", collectionName);

            // 创建默认 Schema
            CollectionSchema schema = CollectionSchema.defaultSchema(properties.getEmbeddingDimension());

            // 创建默认索引
            IndexSchema indexSchema = IndexSchema.builder()
                    .fieldName("embedding")
                    .indexType(properties.getIndexType())
                    .metricType(properties.getMetricType())
                    .build();

            // 创建 Collection
            milvusClient.createCollection(collectionName, schema, indexSchema);

            // 加载 Collection
            milvusClient.loadCollection(collectionName);

            log.info("Default collection initialized: {}", collectionName);
        } else {
            log.info("Collection already exists: {}", collectionName);
        }
    }
}

