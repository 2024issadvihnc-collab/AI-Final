package com.hirematch.service;

import com.hirematch.client.AiServiceClient;
import com.hirematch.dto.ResumeDtos.CreateBuiltResumeRequest;
import com.hirematch.dto.ResumeDtos.ResumeDto;
import com.hirematch.entity.Resume;
import com.hirematch.entity.User;
import com.hirematch.repository.ResumeRepository;
import com.hirematch.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import java.util.*;
import java.util.stream.Collectors;

@Slf4j
@Service
@RequiredArgsConstructor
public class ResumeService {

    private final ResumeRepository resumeRepository;
    private final UserRepository userRepository;
    private final AiServiceClient aiServiceClient;

    @Transactional
    public ResumeDto uploadAndParseResume(String email, MultipartFile file) throws Exception {
        User user = findUser(email);

        Map<String, Object> parsedJson = new HashMap<>();
        String rawText = "";

        // 1. Delegate parsing to AI service
        Optional<Map<String, Object>> aiParsed = aiServiceClient.parseResume(
                file.getBytes(),
                file.getOriginalFilename() != null ? file.getOriginalFilename() : "resume.pdf"
        );

        if (aiParsed.isPresent()) {
            parsedJson = aiParsed.get();
            Object raw = parsedJson.get("raw_text");
            rawText = raw != null ? raw.toString().trim() : "";
        } else {
            log.warn("AI service resume parsing failed, falling back to direct metadata parsing without dummy embeddings");
            parsedJson.put("contact", Map.of("email", user.getEmail()));
            parsedJson.put("summary", "Uploaded Resume: " + file.getOriginalFilename());
            parsedJson.put("skills", List.of("Communication", "Problem Solving"));
            parsedJson.put("experience", List.of());
            parsedJson.put("education", List.of());
            rawText = ""; // Do NOT embed filename!
        }

        // 2. Save in database
        boolean isFirstResume = resumeRepository.findByUserIdOrderByCreatedAtDesc(user.getId()).isEmpty();

        Resume resume = Resume.builder()
                .user(user)
                .version(1)
                .label("Uploaded Resume")
                .isPrimary(isFirstResume)
                .source("UPLOAD")
                .fileName(file.getOriginalFilename())
                .mimeType(file.getContentType())
                .parsedJson(parsedJson)
                .build();

        Resume savedResume = resumeRepository.save(resume);

        // 3. Generate 384-dimensional Sentence-Transformers embedding if raw text is available
        if (!rawText.isBlank()) {
            aiServiceClient.getEmbeddingVectorJson(rawText).ifPresent(vectorJson -> {
                resumeRepository.updateEmbedding(savedResume.getId(), vectorJson);
                log.info("Successfully generated and stored vector embedding for resume {}", savedResume.getId());
            });
        }

        return mapToDto(savedResume);
    }

    @Transactional
    public ResumeDto createBuiltResume(String email, CreateBuiltResumeRequest request) {
        User user = findUser(email);

        boolean isFirstResume = resumeRepository.findByUserIdOrderByCreatedAtDesc(user.getId()).isEmpty();

        Resume resume = Resume.builder()
                .user(user)
                .version(1)
                .label(request.getLabel() != null && !request.getLabel().isBlank() ? request.getLabel() : "Built Resume")
                .isPrimary(isFirstResume)
                .source("BUILDER")
                .fileName(request.getFileName() != null && !request.getFileName().isBlank() ? request.getFileName() : "Resume_AI_Built.pdf")
                .parsedJson(request.getParsedJson())
                .atsScore((short) 85)
                .build();

        Resume savedResume = resumeRepository.save(resume);

        // Generate vector embedding from built resume sections
        String textToEmbed = buildTextFromParsedJson(request.getParsedJson());
        if (!textToEmbed.isBlank()) {
            aiServiceClient.getEmbeddingVectorJson(textToEmbed).ifPresent(vectorJson -> {
                resumeRepository.updateEmbedding(savedResume.getId(), vectorJson);
                log.info("Successfully generated and stored vector embedding for built resume {}", savedResume.getId());
            });
        }

        return mapToDto(savedResume);
    }

    public String buildTextFromParsedJson(Map<String, Object> json) {
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

    @Transactional(readOnly = true)
    public List<ResumeDto> getResumes(String email) {
        User user = findUser(email);
        return resumeRepository.findByUserIdOrderByCreatedAtDesc(user.getId())
                .stream().map(this::mapToDto).collect(Collectors.toList());
    }

    @Transactional
    public ResumeDto setPrimary(String email, Long resumeId) {
        User user = findUser(email);

        Resume resume = resumeRepository.findByIdAndUserId(resumeId, user.getId())
                .orElseThrow(() -> new RuntimeException("Resume not found"));

        resumeRepository.clearPrimaryFlagsForUser(user.getId());
        resume.setIsPrimary(true);
        return mapToDto(resumeRepository.save(resume));
    }

    @Transactional
    public void deleteResume(String email, Long resumeId) {
        User user = findUser(email);
        Resume resume = resumeRepository.findByIdAndUserId(resumeId, user.getId())
                .orElseThrow(() -> new RuntimeException("Resume not found"));
        
        resumeRepository.delete(resume);

        if (resume.getIsPrimary()) {
            resumeRepository.findByUserIdOrderByCreatedAtDesc(user.getId()).stream().findFirst()
                .ifPresent(latest -> {
                    latest.setIsPrimary(true);
                    resumeRepository.save(latest);
                });
        }
    }

    /**
     * Extracts raw text from a resume file by delegating to the AI service.
     */
    public String extractTextOnly(MultipartFile file) throws Exception {
        Optional<Map<String, Object>> parsed = aiServiceClient.parseResume(
                file.getBytes(),
                file.getOriginalFilename() != null ? file.getOriginalFilename() : "resume.pdf"
        );
        if (parsed.isPresent()) {
            Object raw = parsed.get().get("raw_text");
            return raw != null ? raw.toString() : "";
        }
        return "";
    }

    private User findUser(String identifier) {
        if (identifier == null || identifier.isBlank()) {
            throw new RuntimeException("User not found");
        }
        if (identifier.matches("\\d+")) {
            return userRepository.findById(Long.parseLong(identifier))
                    .or(() -> userRepository.findByEmail(identifier))
                    .orElseThrow(() -> new RuntimeException("User not found"));
        }
        return userRepository.findByEmail(identifier)
                .orElseThrow(() -> new RuntimeException("User not found"));
    }

    private ResumeDto mapToDto(Resume resume) {
        return ResumeDto.builder()
                .id(resume.getId())
                .version(resume.getVersion())
                .label(resume.getLabel())
                .isPrimary(resume.getIsPrimary())
                .source(resume.getSource())
                .fileName(resume.getFileName())
                .parsedJson(resume.getParsedJson())
                .atsScore(resume.getAtsScore())
                .createdAt(resume.getCreatedAt())
                .updatedAt(resume.getUpdatedAt())
                .build();
    }
}
