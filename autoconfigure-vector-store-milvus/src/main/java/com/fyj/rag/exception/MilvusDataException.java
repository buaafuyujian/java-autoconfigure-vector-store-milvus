package com.fyj.rag.exception;

/**
 * Milvus 数据操作异常
 */
public class MilvusDataException extends MilvusException {

    public MilvusDataException(String message) {
        super(ErrorCode.UNKNOWN_ERROR, message);
    }

    public MilvusDataException(String message, Throwable cause) {
        super(ErrorCode.UNKNOWN_ERROR, message, cause);
    }

    public MilvusDataException(ErrorCode errorCode, String message) {
        super(errorCode, message);
    }

    public MilvusDataException(ErrorCode errorCode, String message, Throwable cause) {
        super(errorCode, message, cause);
    }
}

