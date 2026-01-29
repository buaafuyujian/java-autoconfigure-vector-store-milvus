package com.fyj.rag.exception;

import lombok.Getter;

/**
 * Milvus 基础异常类
 */
@Getter
public class MilvusException extends RuntimeException {

    private final ErrorCode errorCode;

    public MilvusException(String message) {
        super(message);
        this.errorCode = ErrorCode.UNKNOWN_ERROR;
    }

    public MilvusException(String message, Throwable cause) {
        super(message, cause);
        this.errorCode = ErrorCode.UNKNOWN_ERROR;
    }

    public MilvusException(ErrorCode errorCode) {
        super(errorCode.getMessage());
        this.errorCode = errorCode;
    }

    public MilvusException(ErrorCode errorCode, String message) {
        super(message);
        this.errorCode = errorCode;
    }

    public MilvusException(ErrorCode errorCode, String message, Throwable cause) {
        super(message, cause);
        this.errorCode = errorCode;
    }

    public MilvusException(ErrorCode errorCode, Throwable cause) {
        super(errorCode.getMessage(), cause);
        this.errorCode = errorCode;
    }
}

