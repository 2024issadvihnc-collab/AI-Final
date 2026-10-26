package com.hirematch.service;

import com.hirematch.client.AiServiceClient;
import com.hirematch.config.AppProperties;
import com.hirematch.dto.JobDtos.JobDto;
import com.hirematch.dto.MatchDtos.MatchAnalysisDto;
import com.hirematch.dto.MatchDtos.MatchRecommendationDto;
import com.hirematch.entity.Job;
import com.hirematch.entity.MatchResult;
import com.hirematch.entity.Resume;
import com.hirematch.entity.User;
import com.hirematch.entity.UserProfile;
import com.hirematch.repository.JobRepository;
import com.hirematch.repository.MatchResultRepository;
import com.hirematch.repository.ResumeRepository;
import com.hirematch.repository.UserProfileRepository;
import com.hirematch.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@Slf4j
@Service
@RequiredArgsConstructor
public class MatchService {

    private final JobRepository jobRepository;
    private final ResumeRepository resumeRepository;
    private final UserRepository userRepository;
    private final UserProfileRepository userProfileRepository;
    private final MatchResultRepository matchResultRepository;
    private final AiServiceClient aiServiceClient;
    private final AppProperties appProperties;

    // Baseline transparent defaults for unconfigured experience and location
    private static final double DEFAULT_EXPERIENCE_SCORE = 0.70;
    private static final double DEFAULT_LOCATION_SCORE = 0.80;

    @Transactional
    public MatchAnalysisDto getMatchAnalysis(String email, Long jobId) {
        User user = userRepository.findByEmail(email)
                .orElseThrow(() -> new RuntimeException("User not found with email: " + email));

        Resume primaryResume = resumeRepository.findFirstByUserIdAndIsPrimaryTrue(user.getId())
                .orElseThrow(() -> new RuntimeException("No primary resume found for matching. Please upload or build a resume."));

        Job job = jobRepository.findById(jobId)
                .orElseThrow(() -> new RuntimeException("Job not found with id: " + jobId));

        // Check if primary resume has an embedding; if not, attempt to generate
        ensureResumeEmbedding(primaryResume);

        // Check if job has an embedding; if not, attempt to generate
        ensureJobEmbedding(job);

        // 1. Calculate genuine Semantic Textual Similarity from pgvector
        Double cosineSim = jobRepository.findCosineSimilarity(primaryResume.getId(), jobId).orElse(null);
        Double semanticScore = cosineSim != null ? Math.max(0.0, Math.min(1.0, cosineSim)) : null;

        // 2. Calculate Skill Overlap and identify skill gaps
        List<String> resumeSkills = extractResumeSkills(primaryResume);
        List<String> jobSkills = extractJobSkills(job);

        List<String> matchedSkills = new ArrayList<>();
        List<String> missingSkills = new ArrayList<>();

        for (String js : jobSkills) {
            boolean found = false;
            for (String rs : resumeSkills) {
                if (rs.equalsIgnoreCase(js) || rs.toLowerCase().contains(js.toLowerCase()) || js.toLowerCase().contains(rs.toLowerCase())) {
                    found = true;
                    break;
                }
            }
            if (found) {
                matchedSkills.add(js);
            } else {
                missingSkills.add(js);
            }
        }

        double skillScore = jobSkills.isEmpty() ? 1.0 : (double) matchedSkills.size() / jobSkills.size();

        // 3. Candidate-specific Experience Score
        double experienceScore = computeExperienceScore(primaryResume, job);

        // 4. Candidate-specific Location Score
        double locationScore = computeLocationScore(user, job);

        // 5. Calculate Weighted Overall Score: 0.50 semantic + 0.30 skill + 0.10 experience + 0.10 location
        double wSemantic = appProperties.getMatching().getWeights().getSemantic();
        double wSkill = appProperties.getMatching().getWeights().getSkill();
        double wExperience = appProperties.getMatching().getWeights().getExperience();
        double wLocation = appProperties.getMatching().getWeights().getLocation();

        double effectiveSemantic = semanticScore != null ? semanticScore : 0.0;
        double overallScore = (effectiveSemantic * wSemantic) + (skillScore * wSkill) + (experienceScore * wExperience) + (locationScore * wLocation);

        // Build informative, transparent verdict
        String verdict = buildVerdict(overallScore, semanticScore, matchedSkills, missingSkills);

        // Save or update in cache
        MatchResult result = matchResultRepository.findByResumeIdAndJobId(primaryResume.getId(), jobId)
                .orElse(MatchResult.builder()
                        .resume(primaryResume)
                        .job(job)
                        .build());

        result.setOverallScore((float) (overallScore * 100));
        result.setSemanticScore(semanticScore != null ? (float) (semanticScore * 100) : null);
        result.setSkillScore((float) (skillScore * 100));
        result.setExperienceScore((float) (experienceScore * 100));
        result.setLocationScore((float) (locationScore * 100));
        result.setMatchedSkills(matchedSkills);
        result.setMissingSkills(missingSkills);
        result.setVerdict(verdict);

        return mapToAnalysisDto(matchResultRepository.save(result));
    }

    @Transactional(readOnly = true)
    public List<MatchRecommendationDto> getRecommendations(String email, int limit) {
        if (email == null || email.isBlank()) {
            return getUnmatchedJobRecommendations(limit, "Sign in to see personalized semantic match scores.");
        }

        User user = userRepository.findByEmail(email).orElse(null);
        if (user == null) {
            return getUnmatchedJobRecommendations(limit, "Sign in to see personalized semantic match scores.");
        }

        Resume primaryResume = resumeRepository.findFirstByUserIdAndIsPrimaryTrue(user.getId()).orElse(null);
        if (primaryResume == null) {
            return getUnmatchedJobRecommendations(limit, "Upload or build a primary resume to see AI-powered job matches.");
        }

        Boolean hasEmb = resumeRepository.hasEmbedding(primaryResume.getId());
        if (hasEmb == null || !hasEmb) {
            return getUnmatchedJobRecommendations(limit, "Resume vector embedding pending. Embeddings are required for semantic ranking.");
        }

        // Fetch top matches from pgvector ranked by ascending cosine distance (<=>)
        List<Object[]> rawMatches = jobRepository.findRecommendedJobIdsWithScore(primaryResume.getId(), limit);
        if (rawMatches == null || rawMatches.isEmpty()) {
            return getUnmatchedJobRecommendations(limit, "No job postings with vector embeddings currently match your resume.");
        }

        double wSemantic = appProperties.getMatching().getWeights().getSemantic();
        double wSkill = appProperties.getMatching().getWeights().getSkill();
        double wExperience = appProperties.getMatching().getWeights().getExperience();
        double wLocation = appProperties.getMatching().getWeights().getLocation();

        List<String> resumeSkills = extractResumeSkills(primaryResume);
        List<MatchRecommendationDto> recommendations = new ArrayList<>();

        for (Object[] row : rawMatches) {
            if (row[0] == null || row[1] == null) continue;
            Long jobId = ((Number) row[0]).longValue();
            double similarity = Math.max(0.0, Math.min(1.0, ((Number) row[1]).doubleValue()));

            Job job = jobRepository.findById(jobId).orElse(null);
            if (job == null || !Boolean.TRUE.equals(job.getIsActive())) continue;

            // Skill overlap calculation
            List<String> jobSkills = extractJobSkills(job);
            List<String> matchedSkills = new ArrayList<>();
            List<String> missingSkills = new ArrayList<>();

            for (String js : jobSkills) {
                boolean found = false;
                for (String rs : resumeSkills) {
                    if (rs.equalsIgnoreCase(js) || rs.toLowerCase().contains(js.toLowerCase()) || js.toLowerCase().contains(rs.toLowerCase())) {
                        found = true;
                        break;
                    }
                }
                if (found) {
                    matchedSkills.add(js);
                } else {
                    missingSkills.add(js);
                }
            }

            double skillScore = jobSkills.isEmpty() ? 1.0 : (double) matchedSkills.size() / jobSkills.size();
            double experienceScore = computeExperienceScore(primaryResume, job);
            double locationScore = computeLocationScore(user, job);

            double overallScore = (similarity * wSemantic) + (skillScore * wSkill) + (experienceScore * wExperience) + (locationScore * wLocation);

            String verdict = buildVerdict(overallScore, similarity, matchedSkills, missingSkills);

            recommendations.add(MatchRecommendationDto.builder()
                    .job(mapToJobDto(job))
                    .overallScore((float) Math.round(overallScore * 100))
                    .semanticScore((float) Math.round(similarity * 100))
                    .skillScore((float) Math.round(skillScore * 100))
                    .experienceScore((float) Math.round(experienceScore * 100))
                    .locationScore((float) Math.round(locationScore * 100))
                    .matchedSkills(matchedSkills)
                    .missingSkills(missingSkills)
                    .verdict(verdict)
                    .build());
        }

        // Sort by overallScore descending to respect the 4-component weighted formula
        recommendations.sort((a, b) -> Float.compare(
                b.getOverallScore() != null ? b.getOverallScore() : 0f,
                a.getOverallScore() != null ? a.getOverallScore() : 0f
        ));

        return recommendations;
    }

    /**
     * When semantic matching cannot be run (no resume, unauthenticated, or missing embedding),
     * return active live jobs without inventing fake scores.
     */
    private List<MatchRecommendationDto> getUnmatchedJobRecommendations(int limit, String guidance) {
        List<Job> topJobs = jobRepository.findTop20ByIsActiveTrueOrderByPostedAtDesc();
        List<MatchRecommendationDto> list = new ArrayList<>();
        int count = 0;
        for (Job job : topJobs) {
            if (count >= limit) break;
            list.add(MatchRecommendationDto.builder()
                    .job(mapToJobDto(job))
                    .overallScore(null)
                    .semanticScore(null)
                    .skillScore(null)
                    .experienceScore(null)
                    .locationScore(null)
                    .matchedSkills(Collections.emptyList())
                    .missingSkills(Collections.emptyList())
                    .verdict(guidance)
                    .build());
            count++;
        }
        return list;
    }

    /**
     * Candidate-specific experience matching: compares candidate years of experience
     * against job requirements. If candidate experience cannot be reliably determined,
     * transparently uses a documented neutral baseline.
     */
    private double computeExperienceScore(Resume resume, Job job) {
        if (job.getExperienceMin() == null || job.getExperienceMin() == 0) {
            return 1.0; // Entry-level / no experience requirement
        }

        Integer candidateYears = estimateCandidateExperienceYears(resume);
        if (candidateYears == null) {
            return DEFAULT_EXPERIENCE_SCORE; // Transparent default placeholder
        }

        int min = job.getExperienceMin();
        int max = job.getExperienceMax() != null ? job.getExperienceMax() : min + 3;

        if (candidateYears >= min && candidateYears <= max) {
            return 1.0;
        } else if (candidateYears < min) {
            return Math.max(0.2, (double) candidateYears / min);
        } else {
            return 0.90; // Overqualified or senior
        }
    }

    private Integer estimateCandidateExperienceYears(Resume resume) {
        try {
            if (resume.getParsedJson() != null && resume.getParsedJson().containsKey("experience")) {
                Object expObj = resume.getParsedJson().get("experience");
                if (expObj instanceof List) {
                    List<?> expList = (List<?>) expObj;
                    // Count number of distinct roles as approximate years or check date strings
                    if (!expList.isEmpty()) {
                        return Math.max(1, expList.size() * 2);
                    }
                }
            }
        } catch (Exception e) {
            log.debug("Unable to parse experience years from resume: {}", e.getMessage());
        }
        return null;
    }

    /**
     * Candidate-specific location matching: evaluates remote flexibility or geographic match.
     */
    private double computeLocationScore(User user, Job job) {
        String jobMode = job.getWorkMode() != null ? job.getWorkMode().toUpperCase() : "";
        if ("REMOTE".equals(jobMode)) {
            return 1.0;
        }

        UserProfile profile = userProfileRepository.findByUserId(user.getId()).orElse(null);
        if (profile == null) {
            return DEFAULT_LOCATION_SCORE;
        }

        String prefMode = profile.getPreferredWorkMode() != null ? profile.getPreferredWorkMode().toUpperCase() : "";
        if ("REMOTE".equals(prefMode) && "ONSITE".equals(jobMode)) {
            return 0.40;
        }

        String userLoc = profile.getLocation() != null ? profile.getLocation().toLowerCase().trim() : "";
        String jobLoc = job.getLocation() != null ? job.getLocation().toLowerCase().trim() : "";

        if (userLoc.isEmpty() || jobLoc.isEmpty()) {
            return DEFAULT_LOCATION_SCORE;
        }

        if (userLoc.contains(jobLoc) || jobLoc.contains(userLoc)) {
            return 1.0;
        }

        return 0.50;
    }

    private String buildVerdict(double overallScore, Double semanticScore, List<String> matched, List<String> missing) {
        StringBuilder sb = new StringBuilder();
        if (overallScore >= 0.80) {
            sb.append("Strong Candidate: High semantic alignment and strong technical fit.");
        } else if (overallScore >= 0.65) {
            sb.append("Good Match: Core profile aligns well with key role requirements.");
        } else if (overallScore >= 0.50) {
            sb.append("Moderate Match: Some relevant skills found; tailoring resume is recommended.");
        } else {
            sb.append("Low Match: Significant gaps between resume profile and job description.");
        }

        if (semanticScore != null) {
            sb.append(String.format(" (Semantic similarity: %.0f%%)", semanticScore * 100));
        }

        if (!missing.isEmpty()) {
            sb.append(". Key skill gaps: ").append(String.join(", ", missing.subList(0, Math.min(3, missing.size()))));
        }

        return sb.toString();
    }

    private void ensureResumeEmbedding(Resume resume) {
        Boolean hasEmb = resumeRepository.hasEmbedding(resume.getId());
        if (Boolean.TRUE.equals(hasEmb)) {
            return;
        }

        String textToEmbed = buildResumeTextToEmbed(resume);
        if (textToEmbed != null && !textToEmbed.isBlank()) {
            aiServiceClient.getEmbeddingVectorJson(textToEmbed).ifPresent(vectorJson -> {
                resumeRepository.updateEmbedding(resume.getId(), vectorJson);
                log.info("Generated missing embedding for resume {}", resume.getId());
            });
        }
    }

    private void ensureJobEmbedding(Job job) {
        Boolean hasEmb = jobRepository.hasEmbedding(job.getId());
        if (Boolean.TRUE.equals(hasEmb)) {
            return;
        }

        String textToEmbed = buildJobTextToEmbed(job);
        if (textToEmbed != null && !textToEmbed.isBlank()) {
            aiServiceClient.getEmbeddingVectorJson(textToEmbed).ifPresent(vectorJson -> {
                jobRepository.updateEmbedding(job.getId(), vectorJson);
                log.info("Generated missing embedding for job {}", job.getId());
            });
        }
    }

    public String buildResumeTextToEmbed(Resume resume) {
        Map<String, Object> json = resume.getParsedJson();
        if (json == null || json.isEmpty()) {
            return "";
        }

        StringBuilder sb = new StringBuilder();
        if (json.containsKey("summary") && json.get("summary") != null) {
            sb.append(json.get("summary")).append("\n\n");
        }
        if (json.containsKey("skills") && json.get("skills") instanceof List) {
            List<?> skills = (List<?>) json.get("skills");
            sb.append("Skills: ").append(String.join(", ", skills.stream().map(Object::toString).toList())).append("\n\n");
        }
        if (json.containsKey("experience") && json.get("experience") instanceof List) {
            List<?> exps = (List<?>) json.get("experience");
            sb.append("Experience:\n");
            for (Object item : exps) {
                if (item instanceof Map<?, ?> m) {
                    Object title = m.get("title");
                    Object company = m.get("company");
                    sb.append(title != null ? title : "").append(" at ").append(company != null ? company : "").append(". ");
                    if (m.containsKey("bullets")) sb.append(m.get("bullets")).append("\n");
                    else if (m.containsKey("description")) sb.append(m.get("description")).append("\n");
                }
            }
        }
        if (json.containsKey("projects") && json.get("projects") instanceof List) {
            List<?> projects = (List<?>) json.get("projects");
            sb.append("Projects:\n");
            for (Object item : projects) {
                if (item instanceof Map<?, ?> m) {
                    Object name = m.get("name");
                    Object tech = m.get("tech");
                    Object desc = m.get("description");
                    sb.append(name != null ? name : "").append(": ").append(tech != null ? tech : "").append(" - ").append(desc != null ? desc : "").append("\n");
                }
            }
        }
        if (json.containsKey("raw_text") && json.get("raw_text") != null) {
            String raw = json.get("raw_text").toString().trim();
            if (sb.length() < 100 && !raw.isBlank()) {
                sb.append(raw);
            }
        }
        return sb.toString().trim();
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
            // Strip HTML tags if present
            String cleanDesc = job.getDescription().replaceAll("<[^>]+>", " ").replaceAll("\\s+", " ").trim();
            sb.append(cleanDesc);
        }
        return sb.toString().trim();
    }

    private List<String> extractResumeSkills(Resume resume) {
        try {
            if (resume.getParsedJson() != null && resume.getParsedJson().containsKey("skills")) {
                Object skillsObj = resume.getParsedJson().get("skills");
                if (skillsObj instanceof List) {
                    List<?> list = (List<?>) skillsObj;
                    return list.stream().map(Object::toString).filter(s -> !s.isBlank()).toList();
                }
            }
        } catch (Exception e) {
            log.warn("Failed to parse resume skills for resume {}: {}", resume.getId(), e.getMessage());
        }
        return Collections.emptyList();
    }

    private List<String> extractJobSkills(Job job) {
        String fullText = (job.getTitle() != null ? job.getTitle() : "") + " " +
                          (job.getDescription() != null ? job.getDescription().replaceAll("<[^>]+>", " ") : "");
        List<String> extracted = aiServiceClient.extractSkills(fullText);
        if (!extracted.isEmpty()) {
            return extracted;
        }

        // Fast fallback common skills
        List<String> common = List.of(
                "Java", "Python", "JavaScript", "TypeScript", "React", "Node.js",
                "Spring Boot", "SQL", "PostgreSQL", "MongoDB", "Docker", "Kubernetes",
                "AWS", "GCP", "Azure", "Git", "REST API", "Microservices", "Kafka", "Redis"
        );
        String textLower = fullText.toLowerCase();
        return common.stream().filter(s -> textLower.contains(s.toLowerCase())).toList();
    }

    private MatchAnalysisDto mapToAnalysisDto(MatchResult result) {
        return MatchAnalysisDto.builder()
                .resumeId(result.getResume().getId())
                .jobId(result.getJob().getId())
                .overallScore(result.getOverallScore())
                .semanticScore(result.getSemanticScore())
                .skillScore(result.getSkillScore())
                .experienceScore(result.getExperienceScore())
                .locationScore(result.getLocationScore())
                .matchedSkills(result.getMatchedSkills())
                .missingSkills(result.getMissingSkills())
                .verdict(result.getVerdict())
                .build();
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
