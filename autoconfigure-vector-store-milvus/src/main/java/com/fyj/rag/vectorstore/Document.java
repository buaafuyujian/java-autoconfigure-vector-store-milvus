package com.fyj.rag.vectorstore;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.annotations.SerializedName;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 文档实体
 */
@Data
@SuperBuilder
@NoArgsConstructor
@AllArgsConstructor
public class Document {

    private static final Gson GSON = new Gson();

    /**
     * 文档唯一标识
     */
    private String id;

    /**
     * 文档内容
     */
    private String content;

    /**
     * 向量 embedding
     * 查询时默认不返回，以节省带宽
     */
    @ExcludeField
    private List<Float> embedding;

    /**
     * 元数据
     */
    private Map<String, Object> metadata;

    /**
     * 字段名称常量
     */
    public static final String FIELD_ID = "id";
    public static final String FIELD_CONTENT = "content";
    public static final String FIELD_EMBEDDING = "embedding";
    public static final String FIELD_METADATA = "metadata";

    /**
     * 创建文档
     */
    public static Document of(String id, String content, List<Float> embedding) {
        return Document.builder()
                .id(id)
                .content(content)
                .embedding(embedding)
                .metadata(new HashMap<>())
                .build();
    }

    /**
     * 创建文档（带元数据）
     */
    public static Document of(String id, String content, List<Float> embedding, Map<String, Object> metadata) {
        return Document.builder()
                .id(id)
                .content(content)
                .embedding(embedding)
                .metadata(metadata)
                .build();
    }

    /**
     * 添加元数据
     */
    public Document addMetadata(String key, Object value) {
        if (this.metadata == null) {
            this.metadata = new HashMap<>();
        }
        this.metadata.put(key, value);
        return this;
    }

    /**
     * 获取元数据，确保不为 null
     */
    public Map<String, Object> getMetadata() {
        if (this.metadata == null) {
            this.metadata = new HashMap<>();
        }
        return this.metadata;
    }

    /**
     * 转换为 JsonObject，用于插入 Milvus
     * 使用 Gson 序列化，子类的字段会自动包含
     */
    public JsonObject toJsonObject() {
        return GSON.toJsonTree(this).getAsJsonObject();
    }

    /**
     * 获取指定类需要输出的字段列表（排除带 @ExcludeField 注解的字段）
     * 会递归获取父类的字段
     * <p>
     * 注意：如果字段上有 @SerializedName 注解，则使用注解中的名称
     *
     * @param clazz 文档类型
     * @return 字段名列表（Milvus 中的实际字段名）
     */
    public static List<String> getOutputFields(Class<? extends Document> clazz) {
        List<String> fields = new ArrayList<>();
        Class<?> currentClass = clazz;

        while (currentClass != null && currentClass != Object.class) {
            for (Field field : currentClass.getDeclaredFields()) {
                // 排除静态字段、带 @ExcludeField 注解的字段
                if (Modifier.isStatic(field.getModifiers())) {
                    continue;
                }
                if (field.isAnnotationPresent(ExcludeField.class)) {
                    continue;
                }

                // 获取字段名：优先使用 @SerializedName 注解的值
                String fieldName = getSerializedFieldName(field);

                // 避免重复添加（子类可能覆盖父类字段）
                if (!fields.contains(fieldName)) {
                    fields.add(fieldName);
                }
            }
            currentClass = currentClass.getSuperclass();
        }

        return fields;
    }

    /**
     * 获取字段的序列化名称
     * 如果有 @SerializedName 注解则使用注解值，否则使用字段名
     *
     * @param field 字段
     * @return 序列化名称
     */
    private static String getSerializedFieldName(Field field) {
        SerializedName serializedName = field.getAnnotation(SerializedName.class);
        if (serializedName != null && !serializedName.value().isEmpty()) {
            return serializedName.value();
        }
        return field.getName();
    }
}

