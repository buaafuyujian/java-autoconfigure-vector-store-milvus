# Milvus Vector Store Spring Boot Starter

[![Java](https://img.shields.io/badge/Java-17+-blue.svg)](https://openjdk.java.net/)
[![Spring Boot](https://img.shields.io/badge/Spring%20Boot-3.4.x-green.svg)](https://spring.io/projects/spring-boot)
[![Milvus](https://img.shields.io/badge/Milvus-2.5.x-orange.svg)](https://milvus.io/)

一个用于 Spring Boot 的 Milvus 向量数据库自动配置 Starter，提供简洁易用的 API 来管理向量存储，支持 RAG（检索增强生成）应用场景。

## ✨ 特性

- 🚀 **Spring Boot 自动配置** - 零配置开箱即用
- 🎯 **泛型支持** - 查询和搜索直接返回自定义 Document 子类
- 📦 **分区管理** - 支持按知识库/租户分区存储
- 🔍 **多种搜索方式** - 支持向量搜索、BM25 全文检索、混合搜索
- 🔧 **灵活的 Schema** - 提供流式 API 创建自定义 Collection Schema
- 🤖 **Spring AI 集成** - 可选集成 EmbeddingModel 自动向量化
- 📝 **BM25 全文检索** - 支持基于关键词的全文检索
- ⚡ **混合搜索** - 结合向量语义搜索和 BM25 关键词搜索，可自定义权重

## 📁 项目结构

```
milvus-test/
├── autoconfigure-vector-store-milvus/    # 核心 Starter 模块
│   ├── src/main/java/com/fyj/rag/
│   │   ├── autoconfigure/                # Spring Boot 自动配置
│   │   │   └── MilvusVectorStoreAutoConfiguration.java
│   │   ├── client/                       # Milvus 客户端封装
│   │   │   └── MilvusClient.java
│   │   ├── exception/                    # 自定义异常
│   │   │   ├── MilvusException.java
│   │   │   ├── MilvusCollectionException.java
│   │   │   ├── MilvusSearchException.java
│   │   │   └── ...
│   │   ├── properties/                   # 配置属性
│   │   │   └── MilvusProperties.java
│   │   ├── schema/                       # Schema 定义
│   │   │   ├── CollectionSchema.java
│   │   │   ├── FieldSchema.java
│   │   │   ├── IndexSchema.java
│   │   │   └── FunctionSchema.java       # BM25 Function 定义
│   │   └── vectorstore/                  # 向量存储核心
│   │       ├── MilvusVectorStore.java    # 接口定义
│   │       ├── DefaultMilvusVectorStore.java
│   │       ├── model/
│   │       │   ├── Document.java         # 文档实体基类
│   │       │   └── SearchResult.java
│   │       ├── request/
│   │       │   ├── QueryRequest.java     # 查询请求（泛型）
│   │       │   ├── SearchRequest.java    # 搜索请求（泛型，支持多种搜索类型）
│   │       │   └── SearchType.java       # 搜索类型枚举（VECTOR/BM25/HYBRID）
│   │       └── annotation/
│   │           └── ExcludeField.java     # 排除字段注解
│   └── pom.xml
├── demo/                                  # 示例项目
│   ├── src/
│   │   ├── main/java/com/example/demo/
│   │   │   ├── DemoApplication.java
│   │   │   └── entity/
│   │   │       └── DocumentSegment.java  # 自定义 Document 子类示例
│   │   └── test/java/
│   │       └── DocumentSegmentTests.java # 完整测试用例
│   └── pom.xml
└── README.md
```

## 🔧 快速开始

### 1. 添加依赖

```xml
<dependency>
    <groupId>com.fyj.rag</groupId>
    <artifactId>autoconfigure-vector-store-milvus</artifactId>
    <version>1.2</version>
</dependency>
```

### 2. 配置 Milvus 连接

在 `application.properties` 或 `application.yml` 中配置：

```properties
# Milvus 连接配置
spring.ai.vectorstore.milvus.uri=http://localhost:19530
spring.ai.vectorstore.milvus.database-name=default

# Collection 配置
spring.ai.vectorstore.milvus.collection-name=my_vectors
spring.ai.vectorstore.milvus.embedding-dimension=1536
spring.ai.vectorstore.milvus.metric-type=COSINE
spring.ai.vectorstore.milvus.index-type=AUTOINDEX

# 可选：启动时自动初始化 Collection
spring.ai.vectorstore.milvus.initialize-schema=false

# 可选：认证配置
spring.ai.vectorstore.milvus.token=your-token
# 或使用用户名密码
spring.ai.vectorstore.milvus.username=root
spring.ai.vectorstore.milvus.password=milvus
```

### 3. 使用 VectorStore

```java
@Service
public class VectorService {

    @Autowired
    private MilvusVectorStore vectorStore;

    public void addDocuments(List<Document> documents) {
        vectorStore.add(documents);
    }

    public List<SearchResult<Document>> search(String query, int topK) {
        // 使用泛型 SearchRequest（Lombok Builder）
        SearchRequest<Document> request = SearchRequest.<Document>builder()
            .query(query)
            .topK(topK)
            .build();
        return vectorStore.search(request);
    }
}
```

## 📖 核心概念

### Document 文档实体

`Document` 是向量存储的基本单元，包含以下字段：

```java
public class Document {
    private String id;              // 文档唯一标识
    private String content;         // 文档内容
    private List<Float> embedding;  // 向量（查询时默认不返回）
    private Map<String, Object> metadata;  // 元数据
}
```

### 自定义 Document 子类

你可以继承 `Document` 创建自定义实体，支持 `@SerializedName` 注解映射字段名：

```java
@Data
@SuperBuilder
@NoArgsConstructor
@EqualsAndHashCode(callSuper = true)
public class DocumentSegment extends Document {

    @SerializedName("file_id")  // 映射到 Milvus 中的 file_id 字段
    private String fileId;

    // 创建 Schema
    public static CollectionSchema createSchema(int dimension) {
        return CollectionSchema.create()
                .description("Document segments collection")
                .field(FieldSchema.primaryKeyVarchar("id", 64))
                .field(FieldSchema.varchar("file_id", 64))
                .field(FieldSchema.varchar("content", 65535))
                .field(FieldSchema.floatVector("embedding", dimension))
                .field(FieldSchema.json("metadata"))
                .build();
    }
}
```

### 分区（Partition）

分区用于隔离不同知识库/租户的数据：

```java
// 创建分区
vectorStore.createPartition("knowledge_base_001");

// 添加数据到指定分区
vectorStore.add(documents, "knowledge_base_001");

// 在指定分区搜索（使用泛型 SearchRequest）
SearchRequest<DocumentSegment> request = SearchRequest.<DocumentSegment>builder()
    .query("搜索内容")
    .topK(10)
    .inPartition("knowledge_base_001")
    .documentClass(DocumentSegment.class)
    .build();
vectorStore.search(request);

// 在多个分区搜索（使用 @Singular）
SearchRequest<DocumentSegment> request = SearchRequest.<DocumentSegment>builder()
    .query("搜索内容")
    .topK(10)
    .inPartition("kb_001")
    .inPartition("kb_002")
    .documentClass(DocumentSegment.class)
    .build();
vectorStore.search(request);
```

## 🔍 查询与搜索（泛型 Request）

本项目采用泛型 Builder 模式设计 API（基于 Lombok），类型信息直接封装在 `QueryRequest<T>` 和 `SearchRequest<T>` 中。

### QueryRequest - 条件查询

使用泛型 `QueryRequest<T>` 进行条件查询：

```java
// 方式1: 简单查询（便捷方法）
List<DocumentSegment> segments = vectorStore.query(
    "file_id == 'doc_001'", 
    DocumentSegment.class
);

// 方式2: 使用静态工厂方法
QueryRequest<Document> request = QueryRequest.of("file_id == 'doc_001'");
List<Document> docs = vectorStore.query(request);

// 方式3: 使用 Builder 完整参数
QueryRequest<DocumentSegment> request = QueryRequest.<DocumentSegment>builder()
    .filter("file_id == 'doc_001'")
    .partitionName("partition_kb001")
    .offset(0)
    .limit(100)
    .documentClass(DocumentSegment.class)
    .build();
List<DocumentSegment> segments = vectorStore.query(request);

// 方式4: 使用 of 静态工厂方法（带分区）
QueryRequest<Document> request = QueryRequest.of("file_id == 'doc_001'", "partition_kb001");
List<Document> docs = vectorStore.query(request);

// 根据 ID 获取
List<DocumentSegment> segments = vectorStore.getById(
    Arrays.asList("id1", "id2"),
    DocumentSegment.class
);
```

### SearchRequest - 向量相似度搜索

使用泛型 `SearchRequest<T>` 进行向量搜索，支持三种搜索类型：
- **VECTOR** - 向量相似度搜索（默认）
- **BM25** - BM25 全文检索
- **HYBRID** - 混合搜索（向量 + BM25）

```java
// 方式1: 使用向量搜索（默认）
List<Float> queryVector = embeddingModel.embed("查询文本");
SearchRequest<DocumentSegment> request = SearchRequest.<DocumentSegment>builder()
    .vector(queryVector)
    .topK(10)
    .filter("file_id == 'doc_001'")
    .similarityThreshold(0.7f)
    .documentClass(DocumentSegment.class)
    .build();
List<SearchResult<DocumentSegment>> results = vectorStore.search(request);

// 方式2: 使用 @Singular 添加多个分区
SearchRequest<DocumentSegment> request = SearchRequest.<DocumentSegment>builder()
    .vector(queryVector)
    .topK(10)
    .inPartition("kb_001")     // @Singular 支持多次调用
    .inPartition("kb_002")
    .documentClass(DocumentSegment.class)
    .build();

// 遍历结果
results.forEach(r -> {
    DocumentSegment doc = r.getDocument();  // 直接获取，无需转换
    float score = r.getScore();
    System.out.println(doc.getFileId() + ": " + score);
});
```

### BM25 全文检索

BM25 是一种基于关键词匹配的全文检索算法，适用于精确关键词匹配场景：

```java
// 方式1: 使用 Builder
SearchRequest<DocumentSegment> request = SearchRequest.<DocumentSegment>builder()
    .query("Java 编程 Spring Boot")
    .searchType(SearchType.BM25)
    .topK(10)
    .documentClass(DocumentSegment.class)
    .build();
List<SearchResult<DocumentSegment>> results = vectorStore.search(request);

// 方式2: 使用便捷静态方法
SearchRequest<Document> request = SearchRequest.bm25("人工智能 机器学习", 10);
List<SearchResult<Document>> results = vectorStore.search(request);

// 方式3: 指定文本字段名（默认为 "content"）
SearchRequest<Document> request = SearchRequest.bm25("深度学习", 10, "content");
List<SearchResult<Document>> results = vectorStore.search(request);
```

### 混合搜索（向量 + BM25）

混合搜索结合向量语义搜索和 BM25 关键词搜索，通过加权融合获得更好的搜索效果：

```java
// 方式1: 使用 Builder，自定义权重（向量 70% + BM25 30%）
SearchRequest<DocumentSegment> request = SearchRequest.<DocumentSegment>builder()
    .query("什么是深度学习")
    .searchType(SearchType.HYBRID)
    .vectorWeight(0.7f)    // 向量搜索权重
    .bm25Weight(0.3f)      // BM25 搜索权重
    .topK(10)
    .documentClass(DocumentSegment.class)
    .build();
List<SearchResult<DocumentSegment>> results = vectorStore.search(request);

// 方式2: 使用便捷方法（默认各 50% 权重）
SearchRequest<Document> request = SearchRequest.hybrid("人工智能技术", 10);
List<SearchResult<Document>> results = vectorStore.search(request);

// 方式3: 使用便捷方法，自定义权重
SearchRequest<Document> request = SearchRequest.hybrid("机器学习算法", 10, 0.6f, 0.4f);
List<SearchResult<Document>> results = vectorStore.search(request);
```

### 文本搜索（自动嵌入）

需要配置 `EmbeddingModel`，可直接使用文本进行搜索：

```java
// 创建带 EmbeddingModel 的 VectorStore
MilvusVectorStore vectorStore = milvusClient.getVectorStore(
    collectionName, 
    embeddingModel
);

// 方式1: 使用 Builder（推荐）
SearchRequest<DocumentSegment> request = SearchRequest.<DocumentSegment>builder()
    .query("Spring Boot 框架")
    .topK(10)
    .documentClass(DocumentSegment.class)
    .build();
List<SearchResult<DocumentSegment>> results = vectorStore.search(request);

// 方式2: 在指定分区搜索
SearchRequest<DocumentSegment> request = SearchRequest.<DocumentSegment>builder()
    .query("机器学习算法")
    .topK(10)
    .inPartition("knowledge_base_001")
    .documentClass(DocumentSegment.class)
    .build();

// 方式3: 跨多个分区搜索，带过滤和阈值
SearchRequest<DocumentSegment> request = SearchRequest.<DocumentSegment>builder()
    .query("深度学习模型")
    .topK(10)
    .partitionNames(Arrays.asList("kb_001", "kb_002"))
    .filter("category == 'AI'")
    .similarityThreshold(0.6f)
    .documentClass(DocumentSegment.class)
    .build();
List<SearchResult<DocumentSegment>> results = vectorStore.search(request);

// 默认返回 Document 类型（不指定 documentClass）
SearchRequest<Document> request = SearchRequest.<Document>builder()
    .query("问题")
    .topK(5)
    .build();
List<SearchResult<Document>> results = vectorStore.search(request);
```

## 🏗️ Schema 管理

### 使用 MilvusClient 创建 Collection

```java
@Autowired
private MilvusClient milvusClient;

// 快速创建（使用默认 Schema）
milvusClient.createCollection("my_collection", 1536);

// 使用自定义 Schema
CollectionSchema schema = CollectionSchema.create()
    .description("My custom collection")
    .field(FieldSchema.primaryKeyVarchar("id", 64))
    .field(FieldSchema.varchar("title", 256))
    .field(FieldSchema.varchar("content", 65535))
    .field(FieldSchema.floatVector("embedding", 1536))
    .field(FieldSchema.int64("timestamp"))
    .field(FieldSchema.json("metadata"))
    .enableDynamicField(false)
    .build();

IndexSchema index = IndexSchema.hnsw("embedding", MetricType.COSINE, 16, 256);

milvusClient.createCollection("my_collection", schema, index);
```

### 创建支持 BM25 的 Collection

要支持 BM25 全文检索和混合搜索，需要：
1. 为文本字段启用分词器（`enableAnalyzer`）
2. 添加稀疏向量字段（`sparseFloatVector`）
3. 添加 BM25 Function
4. 为稀疏向量字段创建索引

```java
// 方式1: 使用 Document 的便捷方法
CollectionSchema schema = Document.createSchemaWithBM25(1536);
List<IndexSchema> indexes = Document.createAllIndexes();
milvusClient.createCollection("my_collection", schema, indexes);

// 方式2: 使用自定义 Schema（完整控制）
CollectionSchema schema = CollectionSchema.create()
    .description("Collection with BM25 support")
    .field(FieldSchema.primaryKeyVarchar("id", 64))
    .field(FieldSchema.varcharWithAnalyzer("content", 65535))  // 启用分词器
    .field(FieldSchema.floatVector("embedding", 1536))
    .field(FieldSchema.sparseFloatVector("sparse"))            // 稀疏向量字段
    .field(FieldSchema.json("metadata"))
    .bm25Function("content", "sparse")  // BM25 Function: content -> sparse
    .enableDynamicField(false)
    .build();

// 创建索引（向量索引 + 稀疏向量索引）
List<IndexSchema> indexes = Arrays.asList(
    IndexSchema.autoIndex("embedding", MetricType.COSINE),
    IndexSchema.sparseInvertedIndex("sparse")
);

milvusClient.createCollection("my_collection", schema, indexes);
milvusClient.loadCollection("my_collection");
```

### 索引类型

```java
// ====== 向量索引 ======
// AUTOINDEX（推荐，Milvus 自动选择最佳索引）
IndexSchema.autoIndex("embedding", MetricType.COSINE);

// HNSW（高精度，适合小数据量）
IndexSchema.hnsw("embedding", MetricType.COSINE, 16, 256);

// IVF_FLAT（适合大数据量）
IndexSchema.ivfFlat("embedding", MetricType.COSINE, 1024);

// IVF_SQ8（压缩索引，节省内存）
IndexSchema.ivfSq8("embedding", MetricType.COSINE, 1024);

// ====== 稀疏向量索引（用于 BM25）======
// SPARSE_INVERTED_INDEX（稀疏倒排索引）
IndexSchema.sparseInvertedIndex("sparse");

// SPARSE_WAND（稀疏 WAND 索引，更快的搜索速度）
IndexSchema.sparseWand("sparse");

// 指定 drop_ratio_search 参数
IndexSchema.sparseInvertedIndex("sparse", 0.2);  // 丢弃 20% 的小值
```

## ⚙️ 配置参考

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `spring.ai.vectorstore.milvus.uri` | `http://localhost:19530` | Milvus 服务地址 |
| `spring.ai.vectorstore.milvus.database-name` | `default` | 数据库名称 |
| `spring.ai.vectorstore.milvus.collection-name` | `vector_store` | Collection 名称 |
| `spring.ai.vectorstore.milvus.embedding-dimension` | `1536` | 向量维度 |
| `spring.ai.vectorstore.milvus.metric-type` | `COSINE` | 度量类型 (COSINE, L2, IP) |
| `spring.ai.vectorstore.milvus.index-type` | `AUTOINDEX` | 索引类型 |
| `spring.ai.vectorstore.milvus.initialize-schema` | `false` | 启动时是否自动创建 Collection |
| `spring.ai.vectorstore.milvus.token` | - | 认证 Token |
| `spring.ai.vectorstore.milvus.username` | - | 用户名 |
| `spring.ai.vectorstore.milvus.password` | - | 密码 |
| `spring.ai.vectorstore.milvus.connect-timeout-ms` | `10000` | 连接超时时间（毫秒） |
| `spring.ai.vectorstore.milvus.secure` | `false` | 是否启用 TLS |

## 📋 API 速查

### MilvusVectorStore 接口

```java
// ====== 数据操作 ======
void add(List<Document> documents);
void add(List<Document> documents, String partitionName);
void upsert(List<Document> documents);
void upsert(List<Document> documents, String partitionName);
void delete(List<String> ids);
void delete(List<String> ids, String partitionName);
void deleteByFilter(String filterExpression);
void deleteByFilter(String filterExpression, String partitionName);

// ====== 根据 ID 获取 ======
List<Document> getById(List<String> ids);
<T extends Document> List<T> getById(List<String> ids, Class<T> clazz);
List<Document> getById(List<String> ids, String partitionName);
<T extends Document> List<T> getById(List<String> ids, String partitionName, Class<T> clazz);

// ====== 查询（泛型 Request）======
<T extends Document> List<T> query(QueryRequest<T> request);
List<Document> query(String filterExpression);                    // 便捷方法
<T extends Document> List<T> query(String filterExpression, Class<T> clazz);  // 便捷方法

// ====== 向量搜索（泛型 Request）======
<T extends Document> List<SearchResult<T>> search(SearchRequest<T> request);

// ====== 分区管理 ======
void createPartition(String partitionName);
void dropPartition(String partitionName);
boolean hasPartition(String partitionName);
List<String> listPartitions();
void loadPartition(String partitionName);
void loadPartitions(List<String> partitionNames);
void releasePartition(String partitionName);

// ====== 统计与维护 ======
long count();
long count(String partitionName);
void flush();
void compact();
```

### QueryRequest\<T\> 类

```java
// 静态工厂方法
QueryRequest.of(String filter);                           // 简单查询
QueryRequest.of(String filter, int offset, int limit);    // 带分页
QueryRequest.of(String filter, String partitionName);     // 带分区
QueryRequest.<T>builder();                                // Builder

// Builder 方法（Lombok @Builder 生成）
QueryRequest<DocumentSegment> request = QueryRequest.<DocumentSegment>builder()
    .filter("field == 'value'")          // 过滤表达式
    .partitionName("partition_name")     // 分区名称（可选）
    .offset(0)                           // 偏移量，默认 0
    .limit(100)                          // 限制数量，默认 100
    .outputField("field1")               // @Singular: 添加输出字段
    .outputField("field2")               // 可多次调用
    .documentClass(DocumentSegment.class) // 指定返回类型 ⭐
    .build();
```

### SearchRequest\<T\> 类

```java
// 静态工厂方法
SearchRequest.of(List<Float> vector, int topK);           // 向量搜索
SearchRequest.of(List<Float> vector, int topK, String filter);
SearchRequest.of(String query, int topK);                 // 文本搜索
SearchRequest.bm25(String query, int topK);               // BM25 搜索
SearchRequest.bm25(String query, int topK, String textFieldName);
SearchRequest.hybrid(String query, int topK);             // 混合搜索（默认各 50%）
SearchRequest.hybrid(String query, int topK, float vectorWeight, float bm25Weight);
SearchRequest.<T>builder();                               // Builder

// Builder 方法（Lombok @Builder + @Singular 生成）
SearchRequest<DocumentSegment> request = SearchRequest.<DocumentSegment>builder()
    .query("搜索文本")                       // 文本查询（与 vector 二选一）
    .vector(queryVector)                    // 向量查询（与 query 二选一）
    .searchType(SearchType.VECTOR)          // 搜索类型: VECTOR/BM25/HYBRID
    .vectorFieldName("embedding")           // 向量字段名，默认 "embedding"
    .sparseVectorFieldName("sparse")        // 稀疏向量字段名，默认 "sparse"
    .textFieldName("content")               // 文本字段名，默认 "content"
    .vectorWeight(0.7f)                     // 混合搜索：向量权重，默认 0.5
    .bm25Weight(0.3f)                       // 混合搜索：BM25 权重，默认 0.5
    .topK(10)                               // 返回数量，默认 10
    .filter("field == 'value'")             // 过滤表达式（可选）
    .inPartition("partition1")              // @Singular: 添加分区
    .inPartition("partition2")              // 可多次调用
    .partitionNames(List.of("p1", "p2"))    // 或直接设置列表
    .similarityThreshold(0.7f)              // 相似度阈值，默认 0.0
    .offset(0)                              // 偏移量，默认 0
    .searchParam("nprobe", 10)              // @Singular: 添加搜索参数
    .searchParam("ef", 64)                  // 可多次调用
    .outputField("field1")                  // @Singular: 添加输出字段
    .documentClass(DocumentSegment.class)   // 指定返回类型 ⭐
    .build();
```

### SearchType 枚举

```java
public enum SearchType {
    VECTOR,   // 向量相似度搜索（默认）
    BM25,     // BM25 全文检索
    HYBRID    // 混合搜索（向量 + BM25）
}

// 从字符串转换
SearchType type = SearchType.fromString("bm25");  // 大小写不敏感
SearchType type = SearchType.fromString("unknown", SearchType.VECTOR);  // 带默认值
```

### 核心用法示例

```java
// 查询：类型在 Request 中指定，无需额外传参
QueryRequest<DocumentSegment> qr = QueryRequest.<DocumentSegment>builder()
    .filter("type == 'faq'")
    .documentClass(DocumentSegment.class)
    .build();
List<DocumentSegment> docs = vectorStore.query(qr);

// 向量搜索：类型在 Request 中指定，无需额外传参  
SearchRequest<DocumentSegment> sr = SearchRequest.<DocumentSegment>builder()
    .query("RAG 是什么")
    .topK(5)
    .documentClass(DocumentSegment.class)
    .build();
List<SearchResult<DocumentSegment>> results = vectorStore.search(sr);

// BM25 全文检索
SearchRequest<Document> bm25Req = SearchRequest.bm25("人工智能 机器学习", 10);
List<SearchResult<Document>> bm25Results = vectorStore.search(bm25Req);

// 混合搜索（向量 70% + BM25 30%）
SearchRequest<Document> hybridReq = SearchRequest.hybrid("深度学习技术", 10, 0.7f, 0.3f);
List<SearchResult<Document>> hybridResults = vectorStore.search(hybridReq);
```

### MilvusClient 接口

```java
// ====== Collection 管理 ======
void createCollection(String name, CollectionSchema schema);
void createCollection(String name, CollectionSchema schema, IndexSchema index);
void createCollection(String name, CollectionSchema schema, List<IndexSchema> indexes);  // 支持多索引
void dropCollection(String name);
boolean hasCollection(String name);
List<String> listCollections();

// ====== 获取 VectorStore ======
MilvusVectorStore getVectorStore(String collectionName);
MilvusVectorStore getVectorStore(String collectionName, EmbeddingModel embeddingModel);

// ====== 索引管理 ======
void createIndex(String collectionName, IndexSchema index);
void dropIndex(String collectionName, String fieldName);

// ====== 加载/释放 ======
void loadCollection(String collectionName);
void releaseCollection(String collectionName);
```

## 🧪 运行测试

```bash
cd demo
mvn test -Dtest=DocumentSegmentTests
```

## 📦 依赖版本

| 依赖 | 版本 |
|------|------|
| Java | 17+ |
| Spring Boot | 3.4.5 |
| Milvus SDK | 2.5.8 |
| Spring AI | 1.0.0-M6 (可选) |

## 📝 License

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

