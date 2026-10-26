package com.hirematch.client;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.hirematch.config.AppProperties;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.*;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.stereotype.Component;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.ResourceAccessException;
import org.springframework.web.client.RestClientResponseException;
import org.springframework.web.client.RestTemplate;

import java.util.*;

@Slf4j
@Component
public class AiServiceClient {

    private final RestTemplate restTemplate;
    private final ObjectMapper objectMapper;
    private final String aiServiceUrl;

    public AiServiceClient(
            AppProperties appProperties,
            ObjectMapper objectMapper,
            @Value("${ai.service.url:}") String legacyAiServiceUrl) {
        this.objectMapper = objectMapper;

        // Priority: AppProperties -> legacy @Value -> default
        if (appProperties.getAiService() != null && appProperties.getAiService().getUrl() != null && !appProperties.getAiService().getUrl().isBlank()) {
            this.aiServiceUrl = appProperties.getAiService().getUrl().replaceAll("/+$", "");
        } else if (legacyAiServiceUrl != null && !legacyAiServiceUrl.isBlank()) {
            this.aiServiceUrl = legacyAiServiceUrl.replaceAll("/+$", "");
        } else {
            this.aiServiceUrl = "http://localhost:8000";
        }

        // Configure timeouts: 5s connection, 30s read for Sentence-Transformers
        SimpleClientHttpRequestFactory factory = new SimpleClientHttpRequestFactory();
        factory.setConnectTimeout(5000);
        factory.setReadTimeout(30000);
        this.restTemplate = new RestTemplate(factory);

        log.info("AiServiceClient configured with base URL: {}", this.aiServiceUrl);
    }

    public String getAiServiceUrl() {
        return aiServiceUrl;
    }

    /**
     * Checks if the AI microservice is responsive and whether its embedder is ready.
     */
    public boolean isHealthy() {
        try {
            ResponseEntity<String> resp = restTemplate.getForEntity(aiServiceUrl + "/health", String.class);
            if (resp.getStatusCode().is2xxSuccessful() && resp.getBody() != null) {
                JsonNode root = objectMapper.readTree(resp.getBody());
                JsonNode embedderNode = root.get("embedder");
                if (embedderNode != null && embedderNode.has("status")) {
                    return "loaded".equalsIgnoreCase(embedderNode.get("status").asText());
                }
                return true;
            }
        } catch (Exception e) {
            log.warn("AI service health check failed at {}: {}", aiServiceUrl, e.getMessage());
        }
        return false;
    }

    /**
     * Generates a genuine 384-dimensional vector embedding for the input text.
     * Returns the JSON string representation (e.g. "[0.123, -0.456, ...]")
     * ready to be stored in PostgreSQL pgvector via CAST(? AS vector).
     *
     * Returns Optional.empty() if text is empty, the AI service is unavailable,
     * or the model returned an error.
     */
    public Optional<String> getEmbeddingVectorJson(String text) {
        if (text == null || text.trim().isEmpty()) {
            log.warn("Cannot generate embedding for empty or whitespace-only text");
            return Optional.empty();
        }

        try {
            Map<String, String> request = Map.of("text", text);
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            HttpEntity<Map<String, String>> entity = new HttpEntity<>(request, headers);

            ResponseEntity<String> resp = restTemplate.postForEntity(
                    aiServiceUrl + "/embed", entity, String.class);

            if (resp.getStatusCode().is2xxSuccessful() && resp.getBody() != null) {
                JsonNode root = objectMapper.readTree(resp.getBody());
                JsonNode embeddingNode = root.get("embedding");
                if (embeddingNode != null && embeddingNode.isArray() && embeddingNode.size() == 384) {
                    return Optional.of(embeddingNode.toString());
                } else {
                    log.error("AI service returned invalid embedding node or dimension != 384: {}", embeddingNode);
                }
            }
        } catch (RestClientResponseException e) {
            log.error("AI service /embed returned error HTTP {}: {}", e.getStatusCode(), e.getResponseBodyAsString());
        } catch (ResourceAccessException e) {
            log.error("Failed to connect to AI service at {}: {}", aiServiceUrl, e.getMessage());
        } catch (Exception e) {
            log.error("Unexpected error calling AI service /embed: {}", e.getMessage(), e);
        }

        return Optional.empty();
    }

    /**
     * Extracts skills from the provided text by invoking the AI service /skills/extract.
     */
    public List<String> extractSkills(String text) {
        if (text == null || text.isBlank()) {
            return Collections.emptyList();
        }

        try {
            Map<String, String> request = Map.of("text", text);
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            HttpEntity<Map<String, String>> entity = new HttpEntity<>(request, headers);

            ResponseEntity<String> resp = restTemplate.postForEntity(
                    aiServiceUrl + "/skills/extract", entity, String.class);

            if (resp.getStatusCode().is2xxSuccessful() && resp.getBody() != null) {
                JsonNode root = objectMapper.readTree(resp.getBody());
                JsonNode skillsNode = root.get("skills");
                if (skillsNode != null && skillsNode.isArray()) {
                    List<String> list = new ArrayList<>();
                    skillsNode.forEach(n -> list.add(n.asText()));
                    return list;
                }
            }
        } catch (Exception e) {
            log.warn("Failed to extract skills via AI service ({}): {}", aiServiceUrl, e.getMessage());
        }

        return Collections.emptyList();
    }

    /**
     * Calls /parse-resume with multipart file bytes.
     */
    public Optional<Map<String, Object>> parseResume(byte[] fileBytes, String filename) {
        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);

            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
            ByteArrayResource resource = new ByteArrayResource(fileBytes) {
                @Override
                public String getFilename() {
                    return filename;
                }
            };
            body.add("file", resource);

            HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);
            ResponseEntity<String> response = restTemplate.postForEntity(
                    aiServiceUrl + "/parse-resume", requestEntity, String.class);

            if (response.getStatusCode().is2xxSuccessful() && response.getBody() != null) {
                Map<String, Object> parsed = objectMapper.readValue(response.getBody(), new TypeReference<>() {});
                return Optional.of(parsed);
            }
        } catch (Exception e) {
            log.warn("AI service resume parsing call failed at {}: {}", aiServiceUrl, e.getMessage());
        }

        return Optional.empty();
    }

    /**
     * Calls /enhance for resume ATS optimization.
     */
    public Optional<Map<String, Object>> enhanceResume(Map<String, Object> resumeJson, String jobDescription) {
        try {
            Map<String, Object> req = Map.of(
                    "resume_json", resumeJson != null ? resumeJson : Collections.emptyMap(),
                    "job_description", jobDescription != null ? jobDescription : ""
            );

            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            HttpEntity<Map<String, Object>> entity = new HttpEntity<>(req, headers);

            ResponseEntity<String> resp = restTemplate.postForEntity(
                    aiServiceUrl + "/enhance", entity, String.class);

            if (resp.getStatusCode().is2xxSuccessful() && resp.getBody() != null) {
                Map<String, Object> result = objectMapper.readValue(resp.getBody(), new TypeReference<>() {});
                return Optional.of(result);
            }
        } catch (Exception e) {
            log.error("AI service resume enhancement failed at {}: {}", aiServiceUrl, e.getMessage());
        }

        return Optional.empty();
    }
}
