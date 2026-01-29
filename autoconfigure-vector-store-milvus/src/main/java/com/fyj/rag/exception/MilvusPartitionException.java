package com.fyj.rag.exception;

/**
 * Milvus 分区操作异常
 */
public class MilvusPartitionException extends MilvusException {

    public MilvusPartitionException(String message) {
        super(ErrorCode.UNKNOWN_ERROR, message);
    }

    public MilvusPartitionException(String message, Throwable cause) {
        super(ErrorCode.UNKNOWN_ERROR, message, cause);
    }

    public MilvusPartitionException(ErrorCode errorCode, String message) {
        super(errorCode, message);
    }

    public MilvusPartitionException(ErrorCode errorCode, String message, Throwable cause) {
        super(errorCode, message, cause);
    }
}

