package com.fyj.rag.vectorstore.request;

/**
 * 搜索类型枚举
 * <p>
 * 定义向量搜索的不同模式
 */
public enum SearchType {

    /**
     * 向量相似度搜索（ANN）
     * <p>
     * 基于向量嵌入的近似最近邻搜索，适用于语义相似度匹配
     */
    VECTOR("向量搜索"),

    /**
     * BM25 全文检索
     * <p>
     * 基于 BM25 算法的关键词匹配搜索，适用于精确关键词匹配场景
     */
    BM25("BM25全文检索"),

    /**
     * 混合搜索（Hybrid）
     * <p>
     * 结合向量搜索和 BM25 搜索的混合模式，综合语义相似度和关键词匹配
     */
    HYBRID("混合搜索");

    private final String description;

    SearchType(String description) {
        this.description = description;
    }

    public String getDescription() {
        return description;
    }

    /**
     * 将字符串转换为 SearchType 枚举
     * <p>
     * 支持大小写不敏感匹配，例如 "vector"、"VECTOR"、"Vector" 都可以转换为 VECTOR
     *
     * @param value 字符串值
     * @return 对应的 SearchType 枚举
     * @throws IllegalArgumentException 如果字符串无法匹配任何枚举值
     */
    public static SearchType fromString(String value) {
        if (value == null || value.trim().isEmpty()) {
            throw new IllegalArgumentException("SearchType value cannot be null or empty");
        }
        String upperValue = value.trim().toUpperCase();
        for (SearchType type : SearchType.values()) {
            if (type.name().equals(upperValue)) {
                return type;
            }
        }
        throw new IllegalArgumentException("Unknown SearchType: " + value +
                ". Valid values are: VECTOR, BM25, HYBRID");
    }

    /**
     * 安全地将字符串转换为 SearchType 枚举
     * <p>
     * 如果转换失败，返回默认值而不是抛出异常
     *
     * @param value 字符串值
     * @param defaultType 默认值
     * @return 对应的 SearchType 枚举，如果转换失败则返回默认值
     */
    public static SearchType fromString(String value, SearchType defaultType) {
        try {
            return fromString(value);
        } catch (IllegalArgumentException e) {
            return defaultType;
        }
    }
}

