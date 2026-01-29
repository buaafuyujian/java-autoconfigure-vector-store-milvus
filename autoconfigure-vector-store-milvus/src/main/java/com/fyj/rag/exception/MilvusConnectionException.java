package com.fyj.rag.exception;

/**
 * Milvus 连接异常
 */
public class MilvusConnectionException extends MilvusException {

    public MilvusConnectionException(String message) {
        super(ErrorCode.CONNECTION_FAILED, message);
    }

    public MilvusConnectionException(String message, Throwable cause) {
        super(ErrorCode.CONNECTION_FAILED, message, cause);
    }

    public MilvusConnectionException(ErrorCode errorCode, String message) {
        super(errorCode, message);
    }

    public MilvusConnectionException(ErrorCode errorCode, String message, Throwable cause) {
        super(errorCode, message, cause);
    }
}

