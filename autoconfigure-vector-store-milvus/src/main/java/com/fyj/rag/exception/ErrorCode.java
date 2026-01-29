package com.fyj.rag.exception;

import lombok.Getter;

/**
 * Milvus 错误码枚举
 */
@Getter
public enum ErrorCode {

    // 连接相关 M0xx
    CONNECTION_FAILED("M001", "Failed to connect to Milvus"),
    CONNECTION_TIMEOUT("M002", "Connection timeout"),
    AUTHENTICATION_FAILED("M003", "Authentication failed"),

    // Collection相关 M1xx
    COLLECTION_NOT_FOUND("M101", "Collection not found"),
    COLLECTION_ALREADY_EXISTS("M102", "Collection already exists"),
    COLLECTION_NOT_LOADED("M103", "Collection not loaded"),
    COLLECTION_SCHEMA_INVALID("M104", "Invalid collection schema"),
    COLLECTION_CREATE_FAILED("M105", "Failed to create collection"),
    COLLECTION_DROP_FAILED("M106", "Failed to drop collection"),
    COLLECTION_LOAD_FAILED("M107", "Failed to load collection"),
    COLLECTION_RELEASE_FAILED("M108", "Failed to release collection"),

    // 分区相关 M2xx
    PARTITION_NOT_FOUND("M201", "Partition not found"),
    PARTITION_ALREADY_EXISTS("M202", "Partition already exists"),
    PARTITION_CREATE_FAILED("M203", "Failed to create partition"),
    PARTITION_DROP_FAILED("M204", "Failed to drop partition"),
    PARTITION_LOAD_FAILED("M205", "Failed to load partition"),
    PARTITION_RELEASE_FAILED("M206", "Failed to release partition"),

    // 索引相关 M3xx
    INDEX_CREATE_FAILED("M301", "Failed to create index"),
    INDEX_DROP_FAILED("M302", "Failed to drop index"),
    INDEX_NOT_FOUND("M303", "Index not found"),

    // 数据相关 M4xx
    DATA_INSERT_FAILED("M401", "Data insert failed"),
    DATA_DELETE_FAILED("M402", "Data delete failed"),
    DATA_UPSERT_FAILED("M403", "Data upsert failed"),
    DATA_NOT_FOUND("M404", "Data not found"),
    DATA_INVALID("M405", "Invalid data format"),
    DATA_QUERY_FAILED("M406", "Data query failed"),

    // 搜索相关 M5xx
    SEARCH_FAILED("M501", "Search failed"),
    SEARCH_PARAMS_INVALID("M502", "Invalid search parameters"),

    // 其他 M9xx
    UNKNOWN_ERROR("M999", "Unknown error");

    private final String code;
    private final String message;

    ErrorCode(String code, String message) {
        this.code = code;
        this.message = message;
    }
}

