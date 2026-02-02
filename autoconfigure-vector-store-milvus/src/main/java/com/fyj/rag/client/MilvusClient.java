package com.fyj.rag.client;

import com.fyj.rag.exception.*;
import com.fyj.rag.schema.CollectionSchema;
import com.fyj.rag.schema.FieldSchema;
import com.fyj.rag.schema.FunctionSchema;
import com.fyj.rag.schema.IndexSchema;
import com.fyj.rag.vectorstore.DefaultMilvusVectorStore;
import com.fyj.rag.vectorstore.MilvusVectorStore;
import io.milvus.v2.client.MilvusClientV2;
import io.milvus.v2.common.DataType;
import io.milvus.v2.common.IndexParam;
import io.milvus.v2.service.collection.request.*;
import io.milvus.v2.service.collection.response.DescribeCollectionResp;
import io.milvus.v2.service.collection.response.ListCollectionsResp;
import io.milvus.v2.service.index.request.CreateIndexReq;
import io.milvus.v2.service.index.request.DescribeIndexReq;
import io.milvus.v2.service.index.request.DropIndexReq;
import io.milvus.v2.service.index.response.DescribeIndexResp;
import lombok.extern.slf4j.Slf4j;

import java.io.Closeable;
import java.util.stream.Collectors;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Milvus 客户端包装类
 * <p>
 * 用于管理整个 Milvus 服务，包括 Collection 和索引的管理
 */
@Slf4j
public class MilvusClient implements Closeable {

    private final MilvusClientV2 client;

    public MilvusClient(MilvusClientV2 client) {
        this.client = client;
    }

    // ==================== Collection 管理 ====================

    /**
     * 创建 Collection（使用自定义 Schema）
     */
    public void createCollection(String collectionName, CollectionSchema schema) {
        createCollection(collectionName, schema, (IndexSchema) null);
    }

    /**
     * 创建 Collection（使用自定义 Schema 和索引）
     */
    public void createCollection(String collectionName, CollectionSchema schema, IndexSchema indexSchema) {
        createCollection(collectionName, schema, indexSchema != null ? Collections.singletonList(indexSchema) : null);
    }

    /**
     * 创建 Collection（使用自定义 Schema 和多个索引）
     * <p>
     * 支持同时创建向量索引和稀疏向量索引，用于支持混合搜索
     */
    public void createCollection(String collectionName, CollectionSchema schema, List<IndexSchema> indexSchemas) {
        try {
            // 构建 Schema（包含 Functions）
            CreateCollectionReq.CollectionSchema milvusSchema = buildMilvusSchema(schema);

            CreateCollectionReq.CreateCollectionReqBuilder<?, ?> builder = CreateCollectionReq.builder()
                    .collectionName(collectionName)
                    .collectionSchema(milvusSchema);

            if (schema.getDescription() != null) {
                builder.description(schema.getDescription());
            }

            // 添加索引参数
            if (indexSchemas != null && !indexSchemas.isEmpty()) {
                List<IndexParam> indexParams = indexSchemas.stream()
                        .map(IndexSchema::toIndexParam)
                        .collect(Collectors.toList());
                builder.indexParams(indexParams);
            }


            client.createCollection(builder.build());
            log.info("Created collection: {}", collectionName);
        } catch (Exception e) {
            throw new MilvusCollectionException(ErrorCode.COLLECTION_CREATE_FAILED,
                    "Failed to create collection: " + collectionName, e);
        }
    }

    /**
     * 快速创建 Collection（仅指定维度，使用默认 Schema）
     */
    public void createCollection(String collectionName, int dimension) {
        try {
            client.createCollection(CreateCollectionReq.builder()
                    .collectionName(collectionName)
                    .dimension(dimension)
                    .build());
            log.info("Created collection: {} with dimension: {}", collectionName, dimension);
        } catch (Exception e) {
            throw new MilvusCollectionException(ErrorCode.COLLECTION_CREATE_FAILED,
                    "Failed to create collection: " + collectionName, e);
        }
    }

    /**
     * 快速创建 Collection（指定维度和度量类型）
     */
    public void createCollection(String collectionName, int dimension, IndexParam.MetricType metricType) {
        try {
            client.createCollection(CreateCollectionReq.builder()
                    .collectionName(collectionName)
                    .dimension(dimension)
                    .metricType(metricType.name())
                    .build());
            log.info("Created collection: {} with dimension: {} and metric: {}",
                    collectionName, dimension, metricType);
        } catch (Exception e) {
            throw new MilvusCollectionException(ErrorCode.COLLECTION_CREATE_FAILED,
                    "Failed to create collection: " + collectionName, e);
        }
    }

    /**
     * 删除 Collection
     */
    public void dropCollection(String collectionName) {
        try {
            client.dropCollection(DropCollectionReq.builder()
                    .collectionName(collectionName)
                    .build());
            log.info("Dropped collection: {}", collectionName);
        } catch (Exception e) {
            throw new MilvusCollectionException(ErrorCode.COLLECTION_DROP_FAILED,
                    "Failed to drop collection: " + collectionName, e);
        }
    }

    /**
     * 检查 Collection 是否存在
     */
    public boolean hasCollection(String collectionName) {
        try {
            return client.hasCollection(HasCollectionReq.builder()
                    .collectionName(collectionName)
                    .build());
        } catch (Exception e) {
            throw new MilvusCollectionException(ErrorCode.UNKNOWN_ERROR,
                    "Failed to check collection: " + collectionName, e);
        }
    }

    /**
     * 列出所有 Collection
     */
    public List<String> listCollections() {
        try {
            ListCollectionsResp response = client.listCollections();
            return response.getCollectionNames();
        } catch (Exception e) {
            throw new MilvusCollectionException(ErrorCode.UNKNOWN_ERROR,
                    "Failed to list collections", e);
        }
    }

    /**
     * 获取 Collection 详细信息
     */
    public DescribeCollectionResp describeCollection(String collectionName) {
        try {
            return client.describeCollection(DescribeCollectionReq.builder()
                    .collectionName(collectionName)
                    .build());
        } catch (Exception e) {
            throw new MilvusCollectionException(ErrorCode.COLLECTION_NOT_FOUND,
                    "Failed to describe collection: " + collectionName, e);
        }
    }

    /**
     * 重命名 Collection
     */
    public void renameCollection(String oldName, String newName) {
        try {
            client.renameCollection(RenameCollectionReq.builder()
                    .collectionName(oldName)
                    .newCollectionName(newName)
                    .build());
            log.info("Renamed collection from {} to {}", oldName, newName);
        } catch (Exception e) {
            throw new MilvusCollectionException(ErrorCode.UNKNOWN_ERROR,
                    "Failed to rename collection from " + oldName + " to " + newName, e);
        }
    }

    // ==================== 加载/释放 Collection ====================

    /**
     * 加载 Collection 到内存
     */
    public void loadCollection(String collectionName) {
        try {
            client.loadCollection(LoadCollectionReq.builder()
                    .collectionName(collectionName)
                    .build());
            log.info("Loaded collection: {}", collectionName);
        } catch (Exception e) {
            throw new MilvusCollectionException(ErrorCode.COLLECTION_LOAD_FAILED,
                    "Failed to load collection: " + collectionName, e);
        }
    }

    /**
     * 释放 Collection
     */
    public void releaseCollection(String collectionName) {
        try {
            client.releaseCollection(ReleaseCollectionReq.builder()
                    .collectionName(collectionName)
                    .build());
            log.info("Released collection: {}", collectionName);
        } catch (Exception e) {
            throw new MilvusCollectionException(ErrorCode.COLLECTION_RELEASE_FAILED,
                    "Failed to release collection: " + collectionName, e);
        }
    }

    /**
     * 获取 Collection 加载状态
     */
    public Boolean getLoadState(String collectionName) {
        try {
            return client.getLoadState(GetLoadStateReq.builder()
                    .collectionName(collectionName)
                    .build());
        } catch (Exception e) {
            throw new MilvusCollectionException(ErrorCode.UNKNOWN_ERROR,
                    "Failed to get load state of collection: " + collectionName, e);
        }
    }

    // ==================== 索引管理 ====================

    /**
     * 创建索引
     */
    public void createIndex(String collectionName, IndexSchema indexSchema) {
        try {
            IndexParam indexParam = indexSchema.toIndexParam();
            client.createIndex(CreateIndexReq.builder()
                    .collectionName(collectionName)
                    .indexParams(Collections.singletonList(indexParam))
                    .build());
            log.info("Created index on field: {} in collection: {}",
                    indexSchema.getFieldName(), collectionName);
        } catch (Exception e) {
            throw new MilvusIndexException(ErrorCode.INDEX_CREATE_FAILED,
                    "Failed to create index on field: " + indexSchema.getFieldName(), e);
        }
    }

    /**
     * 删除索引
     */
    public void dropIndex(String collectionName, String fieldName) {
        try {
            client.dropIndex(DropIndexReq.builder()
                    .collectionName(collectionName)
                    .fieldName(fieldName)
                    .build());
            log.info("Dropped index on field: {} in collection: {}", fieldName, collectionName);
        } catch (Exception e) {
            throw new MilvusIndexException(ErrorCode.INDEX_DROP_FAILED,
                    "Failed to drop index on field: " + fieldName, e);
        }
    }

    /**
     * 获取索引信息
     */
    public DescribeIndexResp describeIndex(String collectionName, String fieldName) {
        try {
            return client.describeIndex(DescribeIndexReq.builder()
                    .collectionName(collectionName)
                    .fieldName(fieldName)
                    .build());
        } catch (Exception e) {
            throw new MilvusIndexException(ErrorCode.INDEX_NOT_FOUND,
                    "Failed to describe index on field: " + fieldName, e);
        }
    }

    // ==================== VectorStore 获取 ====================

    /**
     * 获取 VectorStore 实例
     */
    public MilvusVectorStore getVectorStore(String collectionName) {
        return new DefaultMilvusVectorStore(client, collectionName);
    }

    /**
     * 获取 VectorStore 实例（带 EmbeddingModel，支持自动嵌入）
     */
    public MilvusVectorStore getVectorStore(String collectionName,
                                            org.springframework.ai.embedding.EmbeddingModel embeddingModel) {
        return new DefaultMilvusVectorStore(client, collectionName, embeddingModel);
    }


    // ==================== 原始客户端 ====================

    /**
     * 获取原始的 MilvusClientV2 实例
     */
    public MilvusClientV2 getNativeClient() {
        return client;
    }

    // ==================== 资源管理 ====================

    @Override
    public void close() {
        try {
            client.close();
            log.info("Milvus client closed");
        } catch (Exception e) {
            log.error("Failed to close Milvus client", e);
        }
    }

    // ==================== 辅助方法 ====================

    private CreateCollectionReq.CollectionSchema buildMilvusSchema(CollectionSchema schema) {
        CreateCollectionReq.CollectionSchema.CollectionSchemaBuilder<?, ?> builder =
                CreateCollectionReq.CollectionSchema.builder()
                        .enableDynamicField(schema.isEnableDynamicField());

        List<CreateCollectionReq.FieldSchema> fieldSchemas = new ArrayList<>();
        for (FieldSchema field : schema.getFields()) {
            fieldSchemas.add(buildFieldSchema(field));
        }
        builder.fieldSchemaList(fieldSchemas);

        // 添加 Functions（如 BM25）
        if (schema.getFunctions() != null && !schema.getFunctions().isEmpty()) {
            List<CreateCollectionReq.Function> functions = schema.getFunctions().stream()
                    .map(FunctionSchema::toFunction)
                    .collect(Collectors.toList());
            builder.functionList(functions);
        }

        return builder.build();
    }

    private CreateCollectionReq.FieldSchema buildFieldSchema(FieldSchema field) {
        CreateCollectionReq.FieldSchema.FieldSchemaBuilder<?, ?> builder =
                CreateCollectionReq.FieldSchema.builder()
                        .name(field.getName())
                        .dataType(field.getDataType())
                        .isPrimaryKey(field.isPrimaryKey())
                        .autoID(field.isAutoId());

        if (field.getDescription() != null) {
            builder.description(field.getDescription());
        }

        if (field.getMaxLength() != null) {
            builder.maxLength(field.getMaxLength());
        }

        if (field.getDimension() != null) {
            builder.dimension(field.getDimension());
        }

        if (field.getElementType() != null) {
            builder.elementType(field.getElementType());
        }

        if (field.getMaxCapacity() != null) {
            builder.maxCapacity(field.getMaxCapacity());
        }

        // 分词器设置（用于 BM25 全文检索）
        if (field.isEnableAnalyzer()) {
            builder.enableAnalyzer(true);
            if (field.getAnalyzerParams() != null && !field.getAnalyzerParams().isEmpty()) {
                builder.analyzerParams(field.getAnalyzerParams());
            }
        }

        return builder.build();
    }
}

