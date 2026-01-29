package com.fyj.rag.exception;

/**
 * Milvus Collection 操作异常
 */
public class MilvusCollectionException extends MilvusException {

    public MilvusCollectionException(String message) {
        super(ErrorCode.UNKNOWN_ERROR, message);
    }

    public MilvusCollectionException(String message, Throwable cause) {
        super(ErrorCode.UNKNOWN_ERROR, message, cause);
    }

    public MilvusCollectionException(ErrorCode errorCode, String message) {
        super(errorCode, message);
    }

    public MilvusCollectionException(ErrorCode errorCode, String message, Throwable cause) {
        super(errorCode, message, cause);
    }
}

