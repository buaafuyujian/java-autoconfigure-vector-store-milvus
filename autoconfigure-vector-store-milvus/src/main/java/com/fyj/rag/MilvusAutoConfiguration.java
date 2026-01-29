package com.fyj.rag;

import io.milvus.client.MilvusClient;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;

@AutoConfiguration
@ConditionalOnClass(MilvusClient.class)
public class MilvusAutoConfiguration {
}
