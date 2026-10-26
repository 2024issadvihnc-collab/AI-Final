package com.hirematch.controller;

import com.hirematch.dto.JobDtos.JobDto;
import com.hirematch.entity.Job;
import com.hirematch.repository.JobRepository;
import com.hirematch.scheduler.JobIngestionService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;
import java.util.UUID;

@Slf4j
@RestController
@RequestMapping("/api/v1/jobs")
@RequiredArgsConstructor
public class JobController {

    private final JobRepository jobRepository;
    private final JobIngestionService jobIngestionService;

    @GetMapping
    public ResponseEntity<Page<JobDto>> searchJobs(
            @RequestParam(required = false) String query,
            @RequestParam(required = false) String workMode,
            @RequestParam(required = false) String jobType,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size) {
        
        Pageable pageable = PageRequest.of(page, size);
        Page<Job> jobs = jobRepository.searchJobs(query, workMode, jobType, pageable);
        
        return ResponseEntity.ok(jobs.map(this::mapToJobDto));
    }

    @GetMapping("/{id}")
    public ResponseEntity<JobDto> getJobById(@PathVariable Long id) {
        return jobRepository.findById(id)
                .map(job -> ResponseEntity.ok(mapToJobDto(job)))
                .orElse(ResponseEntity.notFound().build());
    }

    @PostMapping("/sync")
    public ResponseEntity<Map<String, Object>> syncJobs() {
        int ingested = jobIngestionService.ingestJobsFromRemotive();
        return ResponseEntity.ok(Map.of("status", "success", "ingested", ingested));
    }

    @PostMapping("/seed")
    public ResponseEntity<String> seedJobs() {
        seedJob("Senior React Developer", "Zepto", "React, TypeScript, Redux, 5+ years experience building scalable frontend applications.");
        seedJob("AI/ML Engineer", "Sarvam AI", "Python, PyTorch, LLMs, Sentence Transformers, PGVector, 3+ years experience.");
        seedJob("Backend Engineer (Java)", "CRED", "Java 21, Spring Boot, Microservices, PostgreSQL, Kafka, Redis. High scale system design.");
        
        return ResponseEntity.ok("Jobs seeded with embeddings.");
    }

    private void seedJob(String title, String company, String description) {
        Job job = Job.builder()
                .title(title)
                .company(company)
                .description(description)
                .source("MANUAL")
                .hash(UUID.randomUUID().toString())
                .location("Bengaluru")
                .workMode("Hybrid")
                .jobType("Full-time")
                .experienceMin((short) 2)
                .experienceMax((short) 5)
                .build();
        
        Job saved = jobRepository.save(job);
        jobIngestionService.generateEmbedding(saved);
    }

    private JobDto mapToJobDto(Job job) {
        return JobDto.builder()
                .id(job.getId())
                .source(job.getSource())
                .externalId(job.getExternalId())
                .title(job.getTitle())
                .company(job.getCompany())
                .location(job.getLocation())
                .workMode(job.getWorkMode())
                .jobType(job.getJobType())
                .experienceMin(job.getExperienceMin())
                .experienceMax(job.getExperienceMax())
                .salaryMin(job.getSalaryMin())
                .salaryMax(job.getSalaryMax())
                .currency(job.getCurrency())
                .description(job.getDescription())
                .applyUrl(job.getApplyUrl())
                .postedAt(job.getPostedAt())
                .expiresAt(job.getExpiresAt())
                .build();
    }
}
