package com.fyj.rag.exception;

/**
 * Milvus 搜索异常
 */
public class MilvusSearchException extends MilvusException {

    public MilvusSearchException(String message) {
        super(ErrorCode.SEARCH_FAILED, message);
    }

    public MilvusSearchException(String message, Throwable cause) {
        super(ErrorCode.SEARCH_FAILED, message, cause);
    }

    public MilvusSearchException(ErrorCode errorCode, String message) {
        super(errorCode, message);
    }

    public MilvusSearchException(ErrorCode errorCode, String message, Throwable cause) {
        super(errorCode, message, cause);
    }
}

