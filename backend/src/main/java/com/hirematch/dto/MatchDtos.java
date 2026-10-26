package com.hirematch.dto;

import com.hirematch.dto.JobDtos.JobDto;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

public class MatchDtos {

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class MatchRecommendationDto {
        private JobDto job;
        private Float overallScore;
        private Float semanticScore;
        private Float skillScore;
        private Float experienceScore;
        private Float locationScore;
        private List<String> matchedSkills;
        private List<String> missingSkills;
        private String verdict;
    }

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class MatchAnalysisDto {
        private Long resumeId;
        private Long jobId;
        private Float overallScore;
        private Float semanticScore;
        private Float skillScore;
        private Float experienceScore;
        private Float locationScore;
        private List<String> matchedSkills;
        private List<String> missingSkills;
        private String verdict;
    }
}
