package com.hirematch.service;

import com.hirematch.client.AiServiceClient;
import com.hirematch.entity.Job;
import com.hirematch.entity.Resume;
import com.hirematch.entity.User;
import com.hirematch.repository.JobRepository;
import com.hirematch.repository.ResumeRepository;
import com.hirematch.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.Map;

@Slf4j
@Service
@RequiredArgsConstructor
public class EnhancementService {

    private final JobRepository jobRepository;
    private final ResumeRepository resumeRepository;
    private final UserRepository userRepository;
    private final AiServiceClient aiServiceClient;

    @Transactional(readOnly = true)
    public Map<String, Object> enhanceResume(String email, Long jobId) {
        User user = userRepository.findByEmail(email)
                .orElseThrow(() -> new RuntimeException("User not found"));

        Resume primaryResume = resumeRepository.findFirstByUserIdAndIsPrimaryTrue(user.getId())
                .orElseThrow(() -> new RuntimeException("No primary resume found for enhancement."));

        Job job = jobRepository.findById(jobId)
                .orElseThrow(() -> new RuntimeException("Job not found"));

        String jobDesc = (job.getTitle() != null ? job.getTitle() : "") + "\n" +
                         (job.getDescription() != null ? job.getDescription() : "");

        return aiServiceClient.enhanceResume(primaryResume.getParsedJson(), jobDesc)
                .orElseThrow(() -> new RuntimeException("Failed to get enhancement response from AI service."));
    }
}
