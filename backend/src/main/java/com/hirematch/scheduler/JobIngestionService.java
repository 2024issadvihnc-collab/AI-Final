package com.hirematch.scheduler;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.hirematch.client.AiServiceClient;
import com.hirematch.entity.Job;
import com.hirematch.repository.JobRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.time.OffsetDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Optional;

@Slf4j
@Service
@RequiredArgsConstructor
public class JobIngestionService {

    private final JobRepository jobRepository;
    private final ObjectMapper objectMapper;
    private final AiServiceClient aiServiceClient;
    private final RestTemplate restTemplate = new RestTemplate();

    // Run every hour.
    @Scheduled(fixedDelay = 3600000, initialDelay = 10000)
    public int ingestJobsFromRemotive() {
        log.info("Starting Remotive job ingestion...");
        try {
            String url = "https://remotive.com/api/remote-jobs?category=software-dev&limit=20";
            ResponseEntity<String> response = restTemplate.getForEntity(url, String.class);
            
            if (response.getStatusCode().is2xxSuccessful()) {
                JsonNode root = objectMapper.readTree(response.getBody());
                JsonNode jobsNode = root.get("jobs");
                
                if (jobsNode != null && jobsNode.isArray()) {
                    int ingested = 0;
                    for (JsonNode jobNode : jobsNode) {
                        if (ingestSingleJob(jobNode)) {
                            ingested++;
                        }
                    }
                    log.info("Successfully ingested {} new jobs from Remotive.", ingested);
                    return ingested;
                }
            }
        } catch (Exception e) {
            log.error("Failed to ingest jobs from Remotive", e);
        }
        return 0;
    }

    private boolean ingestSingleJob(JsonNode jobNode) {
        try {
            String externalId = jobNode.get("id").asText();
            String title = jobNode.get("title").asText();
            String company = jobNode.get("company_name").asText();
            
            // Hash to prevent duplicates
            String hash = calculateHash("REMOTIVE", externalId, title, company);
            Optional<Job> existing = jobRepository.findByHash(hash);
            
            if (existing.isPresent()) {
                return false; // Skip already ingested jobs
            }

            String description = "";
            if (jobNode.has("description")) {
                description = jobNode.get("description").asText().replaceAll("<[^>]+>", " ").replaceAll("\\s+", " ").trim();
            }

            OffsetDateTime postedAt = OffsetDateTime.now();
            if (jobNode.has("publication_date") && !jobNode.get("publication_date").isNull()) {
                String rawDate = jobNode.get("publication_date").asText();
                try {
                    postedAt = OffsetDateTime.parse(rawDate, DateTimeFormatter.ISO_OFFSET_DATE_TIME);
                } catch (Exception ex) {
                    try {
                        postedAt = java.time.LocalDateTime.parse(rawDate, DateTimeFormatter.ISO_LOCAL_DATE_TIME)
                                .atOffset(java.time.ZoneOffset.UTC);
                    } catch (Exception ignored) {
                        postedAt = OffsetDateTime.now();
                    }
                }
            }

            Job job = Job.builder()
                    .source("REMOTIVE")
                    .externalId(externalId)
                    .hash(hash)
                    .title(title)
                    .company(company)
                    .location(jobNode.has("candidate_required_location") ? jobNode.get("candidate_required_location").asText() : "Remote")
                    .workMode("Remote")
                    .jobType(jobNode.has("job_type") ? jobNode.get("job_type").asText() : "Full-time")
                    .salaryMin(BigDecimal.ZERO)
                    .salaryMax(BigDecimal.ZERO)
                    .currency("USD")
                    .description(description)
                    .applyUrl(jobNode.has("url") ? jobNode.get("url").asText() : "")
                    .postedAt(postedAt)
                    .build();

            Job savedJob = jobRepository.save(job);
            
            // Generate Vector Embedding for semantic search
            generateEmbedding(savedJob);
            
            return true;

        } catch (Exception e) {
            log.warn("Error ingesting a single job: {}", e.getMessage());
            return false;
        }
    }

    public void generateEmbedding(Job job) {
        try {
            String textToEmbed = buildJobTextToEmbed(job);
            if (textToEmbed.isBlank()) {
                return;
            }

            aiServiceClient.getEmbeddingVectorJson(textToEmbed).ifPresent(vectorJson -> {
                jobRepository.updateEmbedding(job.getId(), vectorJson);
                log.info("Successfully generated and stored vector embedding for job {}", job.getId());
            });
        } catch (Exception e) {
            log.warn("Failed to generate embedding for job {}", job.getId(), e);
        }
    }

    public String buildJobTextToEmbed(Job job) {
        StringBuilder sb = new StringBuilder();
        sb.append(job.getTitle() != null ? job.getTitle() : "Software Engineer");
        if (job.getCompany() != null && !job.getCompany().isBlank()) {
            sb.append(" at ").append(job.getCompany());
        }
        sb.append(". ");
        if (job.getLocation() != null && !job.getLocation().isBlank()) {
            sb.append("Location: ").append(job.getLocation()).append(". ");
        }
        if (job.getWorkMode() != null && !job.getWorkMode().isBlank()) {
            sb.append("Work Mode: ").append(job.getWorkMode()).append(". ");
        }
        if (job.getDescription() != null && !job.getDescription().isBlank()) {
            String clean = job.getDescription().replaceAll("<[^>]+>", " ").replaceAll("\\s+", " ").trim();
            sb.append(clean);
        }
        return sb.toString().trim();
    }

    private String calculateHash(String source, String externalId, String title, String company) {
        try {
            String raw = source + "|" + externalId + "|" + title + "|" + company;
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] encodedhash = digest.digest(raw.getBytes(StandardCharsets.UTF_8));
            StringBuilder hexString = new StringBuilder(2 * encodedhash.length);
            for (byte b : encodedhash) {
                String hex = Integer.toHexString(0xff & b);
                if (hex.length() == 1) {
                    hexString.append('0');
                }
                hexString.append(hex);
            }
            return hexString.toString();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
