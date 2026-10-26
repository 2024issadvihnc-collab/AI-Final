package com.hirematch.service;

import com.hirematch.client.AiServiceClient;
import com.hirematch.config.AppProperties;
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
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class MatchServiceTest {

    @Mock
    private JobRepository jobRepository;

    @Mock
    private ResumeRepository resumeRepository;

    @Mock
    private UserRepository userRepository;

    @Mock
    private UserProfileRepository userProfileRepository;

    @Mock
    private MatchResultRepository matchResultRepository;

    @Mock
    private AiServiceClient aiServiceClient;

    @Mock
    private AppProperties appProperties;

    @InjectMocks
    private MatchService matchService;

    private User candidateUser;
    private Resume javaResume;
    private Job javaBackendJob;
    private Job reactFrontendJob;
    private Job chefJob;

    @BeforeEach
    void setUp() {
        AppProperties.Matching matching = new AppProperties.Matching();
        AppProperties.Matching.Weights weights = new AppProperties.Matching.Weights();
        weights.setSemantic(0.50);
        weights.setSkill(0.30);
        weights.setExperience(0.10);
        weights.setLocation(0.10);
        matching.setWeights(weights);

        lenient().when(appProperties.getMatching()).thenReturn(matching);

        candidateUser = User.builder()
                .id(1L)
                .email("alex.dev@example.com")
                .role("USER")
                .build();

        Map<String, Object> resumeParsed = new HashMap<>();
        resumeParsed.put("summary", "Senior Java Software Engineer with 5+ years building distributed backend microservices.");
        resumeParsed.put("skills", List.of("Java", "Spring Boot", "PostgreSQL", "Docker", "REST API", "Microservices"));
        resumeParsed.put("experience", List.of(
                Map.of("title", "Senior Java Developer", "company", "Acme Corp", "description", "Built Spring Boot services."),
                Map.of("title", "Software Engineer", "company", "Tech Global", "description", "Developed REST APIs.")
        ));

        javaResume = Resume.builder()
                .id(100L)
                .user(candidateUser)
                .version(1)
                .isPrimary(true)
                .label("Java Backend Resume")
                .parsedJson(resumeParsed)
                .build();

        javaBackendJob = Job.builder()
                .id(201L)
                .title("Senior Java Backend Engineer")
                .company("CRED")
                .location("Bengaluru")
                .workMode("Remote")
                .jobType("Full-time")
                .experienceMin((short) 3)
                .experienceMax((short) 6)
                .description("Build scalable payment microservices using Java 21, Spring Boot, PostgreSQL, Kafka, and Docker.")
                .isActive(true)
                .build();

        reactFrontendJob = Job.builder()
                .id(202L)
                .title("Senior Frontend React Developer")
                .company("Zepto")
                .location("Bengaluru")
                .workMode("Hybrid")
                .jobType("Full-time")
                .experienceMin((short) 3)
                .experienceMax((short) 5)
                .description("Build high-performance web applications using React, TypeScript, Redux, and Tailwind.")
                .isActive(true)
                .build();

        chefJob = Job.builder()
                .id(203L)
                .title("Executive Head Chef")
                .company("Gourmet Kitchen")
                .location("Mumbai")
                .workMode("Onsite")
                .jobType("Full-time")
                .experienceMin((short) 5)
                .experienceMax((short) 10)
                .description("Lead culinary team, manage food cost, design seasonal tasting menus and sourdough baking.")
                .isActive(true)
                .build();
    }

    @Test
    @DisplayName("Overall score accurately implements: 0.50*semantic + 0.30*skill + 0.10*exp + 0.10*loc")
    void testWeightedOverallScoreFormula() {
        when(userRepository.findByEmail(candidateUser.getEmail())).thenReturn(Optional.of(candidateUser));
        when(resumeRepository.findFirstByUserIdAndIsPrimaryTrue(candidateUser.getId())).thenReturn(Optional.of(javaResume));
        when(jobRepository.findById(javaBackendJob.getId())).thenReturn(Optional.of(javaBackendJob));
        when(resumeRepository.hasEmbedding(javaResume.getId())).thenReturn(true);
        when(jobRepository.hasEmbedding(javaBackendJob.getId())).thenReturn(true);

        // Genuine cosine similarity from pgvector (0.80 = 80%)
        when(jobRepository.findCosineSimilarity(javaResume.getId(), javaBackendJob.getId()))
                .thenReturn(Optional.of(0.80));

        // Mock job skills
        when(aiServiceClient.extractSkills(anyString()))
                .thenReturn(List.of("Java", "Spring Boot", "PostgreSQL", "Kafka"));

        when(matchResultRepository.findByResumeIdAndJobId(anyLong(), anyLong())).thenReturn(Optional.empty());
        when(matchResultRepository.save(any(MatchResult.class))).thenAnswer(i -> i.getArgument(0));

        MatchAnalysisDto analysis = matchService.getMatchAnalysis(candidateUser.getEmail(), javaBackendJob.getId());

        assertNotNull(analysis);
        // Semantic: 0.80 -> 80%
        assertEquals(80.0f, analysis.getSemanticScore(), 0.01);

        // Skills: 3 of 4 matched (Java, Spring Boot, PostgreSQL) -> 75%
        assertEquals(75.0f, analysis.getSkillScore(), 0.01);
        assertEquals(List.of("Kafka"), analysis.getMissingSkills());

        // Location: Job is Remote -> 100%
        assertEquals(100.0f, analysis.getLocationScore(), 0.01);

        // Experience: candidate has ~4 years, job min is 3 -> 100%
        assertEquals(100.0f, analysis.getExperienceScore(), 0.01);

        // Overall = (0.80 * 0.50) + (0.75 * 0.30) + (1.0 * 0.10) + (1.0 * 0.10)
        //         = 0.400 + 0.225 + 0.100 + 0.100 = 0.825 -> 82.5%
        assertEquals(82.5f, analysis.getOverallScore(), 0.01);
        assertTrue(analysis.getVerdict().contains("Strong Candidate"));
    }

    @Test
    @DisplayName("Reproducible test: Semantically relevant jobs rank strictly higher than unrelated jobs")
    void testSemanticRankingOrderDifferentRelevance() {
        when(userRepository.findByEmail(candidateUser.getEmail())).thenReturn(Optional.of(candidateUser));
        when(resumeRepository.findFirstByUserIdAndIsPrimaryTrue(candidateUser.getId())).thenReturn(Optional.of(javaResume));
        when(resumeRepository.hasEmbedding(javaResume.getId())).thenReturn(true);

        // Simulate pgvector results with MiniLM cosine similarities:
        // Java Job: 0.82 similarity
        // React Job: 0.45 similarity
        // Chef Job: 0.12 similarity
        List<Object[]> pgVectorResults = List.of(
                new Object[]{201L, 0.82},
                new Object[]{202L, 0.45},
                new Object[]{203L, 0.12}
        );

        when(jobRepository.findRecommendedJobIdsWithScore(eq(javaResume.getId()), anyInt()))
                .thenReturn(pgVectorResults);

        when(jobRepository.findById(201L)).thenReturn(Optional.of(javaBackendJob));
        when(jobRepository.findById(202L)).thenReturn(Optional.of(reactFrontendJob));
        when(jobRepository.findById(203L)).thenReturn(Optional.of(chefJob));

        when(aiServiceClient.extractSkills(contains("Java 21")))
                .thenReturn(List.of("Java", "Spring Boot", "PostgreSQL"));
        when(aiServiceClient.extractSkills(contains("React, TypeScript")))
                .thenReturn(List.of("React", "TypeScript", "Redux"));
        when(aiServiceClient.extractSkills(contains("culinary")))
                .thenReturn(List.of("Culinary Arts", "Menu Design", "Food Safety"));

        List<MatchRecommendationDto> recs = matchService.getRecommendations(candidateUser.getEmail(), 10);

        assertEquals(3, recs.size());

        MatchRecommendationDto first = recs.get(0);
        MatchRecommendationDto second = recs.get(1);
        MatchRecommendationDto third = recs.get(2);

        // Verification 1: Ranking order
        assertEquals("Senior Java Backend Engineer", first.getJob().getTitle());
        assertEquals("Senior Frontend React Developer", second.getJob().getTitle());
        assertEquals("Executive Head Chef", third.getJob().getTitle());

        // Verification 2: Meaningful numeric spread
        assertTrue(first.getOverallScore() > second.getOverallScore(), "Java job score must be > React job score");
        assertTrue(second.getOverallScore() > third.getOverallScore(), "React job score must be > Chef job score");
        assertTrue(first.getSemanticScore() > 80f, "Java job semantic score should be > 80%");
        assertTrue(third.getSemanticScore() < 20f, "Chef job semantic score should be < 20%");
    }

    @Test
    @DisplayName("Handles missing candidate resume gracefully without returning fake random scores")
    void testMissingResumeReturnsUnmatchedGuidance() {
        when(userRepository.findByEmail("noresume@example.com"))
                .thenReturn(Optional.of(User.builder().id(2L).email("noresume@example.com").build()));
        when(resumeRepository.findFirstByUserIdAndIsPrimaryTrue(2L)).thenReturn(Optional.empty());
        when(jobRepository.findTop20ByIsActiveTrueOrderByPostedAtDesc()).thenReturn(List.of(javaBackendJob));

        List<MatchRecommendationDto> recs = matchService.getRecommendations("noresume@example.com", 10);

        assertNotNull(recs);
        assertFalse(recs.isEmpty());
        // Must NOT return fake 94%, 91%, 88% scores!
        assertNull(recs.get(0).getOverallScore(), "Scores must be null when no resume is present");
        assertNull(recs.get(0).getSemanticScore(), "Semantic score must be null when no resume is present");
        assertTrue(recs.get(0).getVerdict().contains("Upload or build a primary resume"));
    }

    @Test
    @DisplayName("Handles missing resume embedding gracefully without fake scores")
    void testMissingResumeEmbeddingReturnsGuidance() {
        when(userRepository.findByEmail(candidateUser.getEmail())).thenReturn(Optional.of(candidateUser));
        when(resumeRepository.findFirstByUserIdAndIsPrimaryTrue(candidateUser.getId())).thenReturn(Optional.of(javaResume));
        when(resumeRepository.hasEmbedding(javaResume.getId())).thenReturn(false);
        when(jobRepository.findTop20ByIsActiveTrueOrderByPostedAtDesc()).thenReturn(List.of(javaBackendJob));

        List<MatchRecommendationDto> recs = matchService.getRecommendations(candidateUser.getEmail(), 10);

        assertNotNull(recs);
        assertNull(recs.get(0).getOverallScore());
        assertTrue(recs.get(0).getVerdict().contains("Resume vector embedding pending"));
    }

    @Test
    @DisplayName("Attempts to generate missing job embedding when computing detailed match analysis")
    void testGeneratesMissingJobEmbeddingOnTheFly() {
        when(userRepository.findByEmail(candidateUser.getEmail())).thenReturn(Optional.of(candidateUser));
        when(resumeRepository.findFirstByUserIdAndIsPrimaryTrue(candidateUser.getId())).thenReturn(Optional.of(javaResume));
        when(jobRepository.findById(javaBackendJob.getId())).thenReturn(Optional.of(javaBackendJob));
        when(resumeRepository.hasEmbedding(javaResume.getId())).thenReturn(true);

        // Job currently has no embedding
        when(jobRepository.hasEmbedding(javaBackendJob.getId())).thenReturn(false);
        when(aiServiceClient.getEmbeddingVectorJson(anyString())).thenReturn(Optional.of("[0.01, 0.02, 0.03]"));

        // After generation, similarity can be computed
        when(jobRepository.findCosineSimilarity(javaResume.getId(), javaBackendJob.getId()))
                .thenReturn(Optional.of(0.78));

        when(matchResultRepository.findByResumeIdAndJobId(anyLong(), anyLong())).thenReturn(Optional.empty());
        when(matchResultRepository.save(any())).thenAnswer(i -> i.getArgument(0));

        MatchAnalysisDto analysis = matchService.getMatchAnalysis(candidateUser.getEmail(), javaBackendJob.getId());

        // Verifies jobRepository.updateEmbedding was called to store the newly computed vector
        verify(jobRepository, times(1)).updateEmbedding(eq(javaBackendJob.getId()), eq("[0.01, 0.02, 0.03]"));
        assertEquals(78.0f, analysis.getSemanticScore(), 0.01);
    }
}
