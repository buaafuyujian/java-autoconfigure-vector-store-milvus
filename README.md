# Milvus Vector Store Spring Boot Starter

[![Java](https://img.shields.io/badge/Java-17+-blue.svg)](https://openjdk.java.net/)
[![Spring Boot](https://img.shields.io/badge/Spring%20Boot-3.4.x-green.svg)](https://spring.io/projects/spring-boot)
[![Milvus](https://img.shields.io/badge/Milvus-2.5.x-orange.svg)](https://milvus.io/)

一个用于 Spring Boot 的 Milvus 向量数据库自动配置 Starter，提供简洁易用的 API 来管理向量存储，支持 RAG（检索增强生成）应用场景。

## ✨ 特性

- 🚀 **Spring Boot 自动配置** - 零配置开箱即用
- 🎯 **泛型支持** - 查询和搜索直接返回自定义 Document 子类
- 📦 **分区管理** - 支持按知识库/租户分区存储
- 🔍 **多种搜索方式** - 支持向量搜索、文本搜索（自动嵌入）、过滤查询
- 🔧 **灵活的 Schema** - 提供流式 API 创建自定义 Collection Schema
- 🤖 **Spring AI 集成** - 可选集成 EmbeddingModel 自动向量化

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
│   │   │   └── IndexSchema.java
│   │   └── vectorstore/                  # 向量存储核心
│   │       ├── MilvusVectorStore.java    # 接口定义
│   │       ├── DefaultMilvusVectorStore.java
│   │       ├── Document.java             # 文档实体基类
│   │       ├── QueryRequest.java         # 查询请求（Spring AI 风格）
│   │       ├── SearchRequest.java        # 搜索请求（Spring AI 风格）
│   │       ├── SearchResult.java
│   │       └── ExcludeField.java         # 排除字段注解
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

    public List<SearchResult> search(String query, int topK) {
        // 使用 Spring AI 风格的 SearchRequest
        SearchRequest request = SearchRequest.builder()
            .query(query)
            .topK(topK)
            .build();
        return vectorStore.similaritySearch(request);
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

// 在指定分区搜索（使用 SearchRequest）
SearchRequest request = SearchRequest.builder()
    .query("搜索内容")
    .topK(10)
    .partitionNames(Collections.singletonList("knowledge_base_001"))
    .build();
vectorStore.similaritySearch(request, DocumentSegment.class);

// 在多个分区搜索
SearchRequest request = SearchRequest.builder()
    .query("搜索内容")
    .topK(10)
    .partitionNames(Arrays.asList("kb_001", "kb_002"))
    .build();
vectorStore.similaritySearch(request, DocumentSegment.class);
```

## 🔍 查询与搜索（Spring AI 风格）

本项目采用 Spring AI 风格的 Builder 模式设计 API，使用 `QueryRequest` 和 `SearchRequest` 封装请求参数，避免大量方法重载，提供更好的可读性和扩展性。

### QueryRequest - 条件查询

使用 `QueryRequest` 进行条件查询：

```java
// 方式1: 简单查询（便捷方法）
List<DocumentSegment> segments = vectorStore.query(
    "file_id == 'doc_001'", 
    DocumentSegment.class
);

// 方式2: 使用静态工厂方法
QueryRequest request = QueryRequest.filter("file_id == 'doc_001'");
List<DocumentSegment> segments = vectorStore.query(request, DocumentSegment.class);

// 方式3: 使用 Builder 完整参数
QueryRequest request = QueryRequest.builder()
    .filterExpression("file_id == 'doc_001'")
    .partitionName("partition_kb001")
    .offset(0)
    .limit(100)
    .build();
List<DocumentSegment> segments = vectorStore.query(request, DocumentSegment.class);

// 方式4: 使用静态工厂方法（带分区）
QueryRequest request = QueryRequest.inPartition("file_id == 'doc_001'", "partition_kb001");
List<DocumentSegment> segments = vectorStore.query(request, DocumentSegment.class);

// 根据 ID 获取
List<DocumentSegment> segments = vectorStore.getById(
    Arrays.asList("id1", "id2"),
    DocumentSegment.class
);
```

### SearchRequest - 向量相似度搜索

使用 `SearchRequest` 进行向量搜索，支持向量查询和文本查询：

```java
// 方式1: 使用向量搜索
List<Float> queryVector = embeddingModel.embed("查询文本");
SearchRequest request = SearchRequest.builder()
    .vector(queryVector)
    .topK(10)
    .filter("file_id == 'doc_001'")
    .similarityThreshold(0.7f)
    .build();
List<SearchResult<DocumentSegment>> results = vectorStore.similaritySearch(request, DocumentSegment.class);

// 方式2: 使用静态工厂方法
SearchRequest request = SearchRequest.vector(queryVector)
    .topK(10)
    .filter("category == 'tech'")
    .build();

// 方式3: 指定分区搜索
SearchRequest request = SearchRequest.builder()
    .vector(queryVector)
    .topK(10)
    .partitionNames(List.of("kb_001", "kb_002"))
    .build();

// 遍历结果
results.forEach(r -> {
    DocumentSegment doc = r.getDocument();  // 直接获取，无需转换
    float score = r.getScore();
    System.out.println(doc.getFileId() + ": " + score);
});
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
SearchRequest request = SearchRequest.builder()
    .query("Spring Boot 框架")  // 直接传入文本
    .topK(10)
    .build();
List<SearchResult<DocumentSegment>> results = vectorStore.similaritySearch(request, DocumentSegment.class);

// 方式2: 使用静态工厂方法
SearchRequest request = SearchRequest.query("人工智能技术")
    .topK(5)
    .build();

// 方式3: 在指定分区搜索
SearchRequest request = SearchRequest.builder()
    .query("机器学习算法")
    .topK(10)
    .partitionNames(Collections.singletonList("knowledge_base_001"))
    .build();

// 方式4: 跨多个分区搜索
SearchRequest request = SearchRequest.builder()
    .query("深度学习模型")
    .topK(10)
    .partitionNames(Arrays.asList("kb_001", "kb_002"))
    .filter("category == 'AI'")
    .similarityThreshold(0.6f)
    .build();
List<SearchResult<DocumentSegment>> results = vectorStore.similaritySearch(request, DocumentSegment.class);
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

### 索引类型

```java
// AUTOINDEX（推荐，Milvus 自动选择最佳索引）
IndexSchema.autoIndex("embedding", MetricType.COSINE);

// HNSW（高精度，适合小数据量）
IndexSchema.hnsw("embedding", MetricType.COSINE, 16, 256);

// IVF_FLAT（适合大数据量）
IndexSchema.ivfFlat("embedding", MetricType.COSINE, 1024);

// IVF_SQ8（压缩索引，节省内存）
IndexSchema.ivfSq8("embedding", MetricType.COSINE, 1024);
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

// ====== 查询（Spring AI 风格）======
List<Document> query(QueryRequest request);
<T extends Document> List<T> query(QueryRequest request, Class<T> clazz);
List<Document> query(String filterExpression);                    // 便捷方法
<T extends Document> List<T> query(String filterExpression, Class<T> clazz);  // 便捷方法

// ====== 向量搜索（Spring AI 风格）======
List<SearchResult> similaritySearch(SearchRequest request);
<T extends Document> List<SearchResult<T>> similaritySearch(SearchRequest request, Class<T> clazz);

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

### QueryRequest 类

```java
// 静态工厂方法
QueryRequest.filter(String filterExpression);
QueryRequest.of(String filterExpression, int offset, int limit);
QueryRequest.inPartition(String filterExpression, String partitionName);

// Builder 方式
QueryRequest request = QueryRequest.builder()
    .filterExpression("field == 'value'")  // 过滤表达式
    .partitionName("partition_name")        // 分区名称（可选）
    .offset(0)                              // 偏移量，默认 0
    .limit(100)                             // 限制数量，默认 100
    .outputFields(List.of("field1"))        // 输出字段（可选）
    .build();
```

### SearchRequest 类

```java
// 静态工厂方法
SearchRequest.vector(List<Float> vector);   // 向量搜索
SearchRequest.query(String query);          // 文本搜索
SearchRequest.of(List<Float> vector, int topK);
SearchRequest.of(List<Float> vector, int topK, String filter);
SearchRequest.of(String query, int topK);
SearchRequest.of(String query, int topK, String filter);

// Builder 方式
SearchRequest request = SearchRequest.builder()
    .query("搜索文本")                       // 文本查询（与 vector 二选一）
    .vector(queryVector)                    // 向量查询（与 query 二选一）
    .vectorFieldName("embedding")           // 向量字段名，默认 "embedding"
    .topK(10)                               // 返回数量，默认 10
    .filter("field == 'value'")             // 过滤表达式（可选）
    .partitionNames(List.of("p1", "p2"))    // 分区列表（可选）
    .similarityThreshold(0.7f)              // 相似度阈值，默认 0.0
    .offset(0)                              // 偏移量，默认 0
    .searchParams(Map.of("nprobe", 10))     // 搜索参数（可选）
    .outputFields(List.of("field1"))        // 输出字段（可选）
    .build();

// 链式调用方法
request.nprobe(10);                         // 设置 IVF 索引的 nprobe
request.ef(64);                             // 设置 HNSW 索引的 ef
request.inPartition("partition_name");      // 添加分区
request.inPartitions(List.of("p1", "p2"));  // 设置多个分区
```

### MilvusClient 接口

```java
// ====== Collection 管理 ======
void createCollection(String name, CollectionSchema schema);
void createCollection(String name, CollectionSchema schema, IndexSchema index);
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

