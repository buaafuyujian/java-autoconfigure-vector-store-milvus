package com.fyj.rag.exception;

/**
 * Milvus 索引操作异常
 */
public class MilvusIndexException extends MilvusException {

    public MilvusIndexException(String message) {
        super(ErrorCode.UNKNOWN_ERROR, message);
    }

    public MilvusIndexException(String message, Throwable cause) {
        super(ErrorCode.UNKNOWN_ERROR, message, cause);
    }

    public MilvusIndexException(ErrorCode errorCode, String message) {
        super(errorCode, message);
    }

    public MilvusIndexException(ErrorCode errorCode, String message, Throwable cause) {
        super(errorCode, message, cause);
    }
}

