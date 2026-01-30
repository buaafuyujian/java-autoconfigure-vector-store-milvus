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
│   │       ├── SearchRequest.java
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
    <version>1.1</version>
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
        return vectorStore.similaritySearch(query, topK);
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

// 在指定分区搜索
vectorStore.similaritySearchInPartition(query, topK, "knowledge_base_001");
```

## 🔍 查询与搜索

### 泛型查询（推荐）

直接返回自定义 Document 子类，无需手动转换：

```java
// 按条件查询，直接返回 DocumentSegment 类型
List<DocumentSegment> segments = vectorStore.query(
    "file_id == 'doc_001'", 
    DocumentSegment.class
);

// 分区查询带分页
List<DocumentSegment> segments = vectorStore.query(
    "file_id == 'doc_001'",
    "partition_kb001",
    0, 100,
    DocumentSegment.class
);

// 根据 ID 获取
List<DocumentSegment> segments = vectorStore.getById(
    Arrays.asList("id1", "id2"),
    DocumentSegment.class
);
```

### 向量相似度搜索

```java
// 使用向量搜索
List<Float> queryVector = embeddingModel.embed("查询文本");
List<SearchResult<DocumentSegment>> results = vectorStore.similaritySearch(
    SearchRequest.builder()
        .vector(queryVector)
        .topK(10)
        .filter("file_id == 'doc_001'")
        .similarityThreshold(0.7f)
        .build(),
    DocumentSegment.class
);

// 遍历结果
results.forEach(r -> {
    DocumentSegment doc = r.getDocument();  // 直接获取，无需转换
    float score = r.getScore();
    System.out.println(doc.getFileId() + ": " + score);
});
```

### 文本搜索（自动嵌入）

需要配置 `EmbeddingModel`：

```java
// 创建带 EmbeddingModel 的 VectorStore
MilvusVectorStore vectorStore = milvusClient.getVectorStore(
    collectionName, 
    embeddingModel
);

// 直接使用文本搜索，自动转换为向量
List<SearchResult<DocumentSegment>> results = vectorStore.similaritySearch(
    "Spring Boot 框架", 
    10, 
    DocumentSegment.class
);

// 在指定分区搜索
List<SearchResult<DocumentSegment>> results = vectorStore.similaritySearchInPartition(
    "人工智能技术",
    5,
    "knowledge_base_001",
    DocumentSegment.class
);

// 跨多个分区搜索
List<SearchResult<DocumentSegment>> results = vectorStore.similaritySearchInPartitions(
    "机器学习算法",
    10,
    Arrays.asList("kb_001", "kb_002"),
    DocumentSegment.class
);
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
void delete(List<String> ids);
void deleteByFilter(String filterExpression);

// ====== 查询 ======
List<T> getById(List<String> ids, Class<T> clazz);
List<T> query(String filterExpression, Class<T> clazz);
List<T> query(String filterExpression, String partitionName, int offset, int limit, Class<T> clazz);

// ====== 向量搜索 ======
List<SearchResult<T>> similaritySearch(SearchRequest request, Class<T> clazz);
List<SearchResult<T>> similaritySearch(String query, int topK, Class<T> clazz);
List<SearchResult<T>> similaritySearchInPartition(String query, int topK, String partitionName, Class<T> clazz);
List<SearchResult<T>> similaritySearchInPartitions(String query, int topK, List<String> partitionNames, Class<T> clazz);

// ====== 分区管理 ======
void createPartition(String partitionName);
void dropPartition(String partitionName);
boolean hasPartition(String partitionName);
List<String> listPartitions();

// ====== 统计 ======
long count();
long count(String partitionName);
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

