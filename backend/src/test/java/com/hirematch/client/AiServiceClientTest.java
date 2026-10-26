package com.hirematch.client;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.hirematch.config.AppProperties;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.test.util.ReflectionTestUtils;
import org.springframework.test.web.client.MockRestServiceServer;
import org.springframework.web.client.RestTemplate;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.*;
import static org.springframework.test.web.client.response.MockRestResponseCreators.*;

class AiServiceClientTest {

    private AiServiceClient aiServiceClient;
    private MockRestServiceServer mockServer;
    private ObjectMapper objectMapper;

    @BeforeEach
    void setUp() {
        objectMapper = new ObjectMapper();
        AppProperties properties = new AppProperties();
        properties.getAiService().setUrl("http://localhost:8000");

        aiServiceClient = new AiServiceClient(properties, objectMapper, null);

        RestTemplate restTemplate = (RestTemplate) ReflectionTestUtils.getField(aiServiceClient, "restTemplate");
        assertNotNull(restTemplate);
        mockServer = MockRestServiceServer.createServer(restTemplate);
    }

    @Test
    @DisplayName("Successfully receives and validates 384-dimensional vector from AI service")
    void testGetEmbeddingSuccess() {
        List<Double> mockVector = new ArrayList<>();
        for (int i = 0; i < 384; i++) {
            mockVector.add(0.01 * (i % 10));
        }
        String mockResponse = "{\"embedding\":" + mockVector.toString() + ",\"dim\":384,\"model\":\"sentence-transformers/all-MiniLM-L6-v2\"}";

        mockServer.expect(requestTo("http://localhost:8000/embed"))
                .andExpect(method(HttpMethod.POST))
                .andExpect(content().contentType(MediaType.APPLICATION_JSON))
                .andRespond(withSuccess(mockResponse, MediaType.APPLICATION_JSON));

        Optional<String> vectorJson = aiServiceClient.getEmbeddingVectorJson("Senior Java Developer with Spring Boot experience");

        mockServer.verify();
        assertTrue(vectorJson.isPresent());
        assertTrue(vectorJson.get().startsWith("["));
        assertTrue(vectorJson.get().endsWith("]"));
    }

    @Test
    @DisplayName("Rejects empty or whitespace-only text before sending network request")
    void testGetEmbeddingRejectsEmptyText() {
        Optional<String> emptyResult = aiServiceClient.getEmbeddingVectorJson("");
        assertFalse(emptyResult.isPresent());

        Optional<String> spacesResult = aiServiceClient.getEmbeddingVectorJson("    \n\t ");
        assertFalse(spacesResult.isPresent());

        Optional<String> nullResult = aiServiceClient.getEmbeddingVectorJson(null);
        assertFalse(nullResult.isPresent());
    }

    @Test
    @DisplayName("Gracefully handles 503 Service Unavailable when AI model is not ready")
    void testGetEmbeddingHandles503Gracefully() {
        String errorJson = "{\"error\":\"Model not loaded. ENABLE_LOCAL_EMBEDDER must be true.\",\"status\":\"disabled\"}";

        mockServer.expect(requestTo("http://localhost:8000/embed"))
                .andExpect(method(HttpMethod.POST))
                .andRespond(withStatus(HttpStatus.SERVICE_UNAVAILABLE)
                        .contentType(MediaType.APPLICATION_JSON)
                        .body(errorJson));

        Optional<String> vectorJson = aiServiceClient.getEmbeddingVectorJson("Valid text");

        mockServer.verify();
        assertFalse(vectorJson.isPresent(), "Must return empty optional on 503 instead of crashing");
    }

    @Test
    @DisplayName("Health check accurately reports when model is loaded")
    void testIsHealthyReportsTrueWhenLoaded() {
        String healthJson = "{\"status\":\"ok\",\"embedder\":{\"enabled\":true,\"status\":\"loaded\",\"dimension\":384}}";

        mockServer.expect(requestTo("http://localhost:8000/health"))
                .andExpect(method(HttpMethod.GET))
                .andRespond(withSuccess(healthJson, MediaType.APPLICATION_JSON));

        boolean healthy = aiServiceClient.isHealthy();
        mockServer.verify();
        assertTrue(healthy);
    }
}
