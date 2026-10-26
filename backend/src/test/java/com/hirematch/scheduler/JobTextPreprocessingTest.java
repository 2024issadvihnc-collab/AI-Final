package com.hirematch.scheduler;

import com.hirematch.entity.Job;
import com.hirematch.entity.Resume;
import com.hirematch.service.ResumeService;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class JobTextPreprocessingTest {

    @Test
    @DisplayName("Job ingestion preprocessing strips HTML tags and formats unified text for embedding")
    void testBuildJobTextToEmbedStripsHtml() {
        JobIngestionService service = new JobIngestionService(null, null, null);

        Job job = Job.builder()
                .title("Full Stack Developer")
                .company("FinTech Innovations")
                .location("Remote")
                .workMode("Remote")
                .description("<p>Looking for a <strong>Java & React</strong> engineer.</p><ul><li>Spring Boot</li><li>PostgreSQL</li></ul>")
                .build();

        String embeddedText = service.buildJobTextToEmbed(job);

        assertNotNull(embeddedText);
        assertFalse(embeddedText.contains("<p>"));
        assertFalse(embeddedText.contains("<strong>"));
        assertFalse(embeddedText.contains("</li>"));
        assertTrue(embeddedText.contains("Full Stack Developer at FinTech Innovations"));
        assertTrue(embeddedText.contains("Location: Remote"));
        assertTrue(embeddedText.contains("Spring Boot"));
    }

    @Test
    @DisplayName("Resume builder text generation formats skills, experience, and summary into unified text")
    void testBuildResumeTextFromParsedJson() {
        ResumeService resumeService = new ResumeService(null, null, null);

        Map<String, Object> parsedJson = Map.of(
                "summary", "Experienced Cloud and DevOps Engineer.",
                "skills", List.of("Kubernetes", "Docker", "AWS", "Terraform"),
                "experience", List.of(
                        Map.of("title", "DevOps Engineer", "company", "Cloudify", "description", "Automated CI/CD pipelines.")
                )
        );

        String text = resumeService.buildTextFromParsedJson(parsedJson);

        assertNotNull(text);
        assertTrue(text.contains("Experienced Cloud and DevOps Engineer"));
        assertTrue(text.contains("Kubernetes, Docker, AWS, Terraform"));
        assertTrue(text.contains("DevOps Engineer at Cloudify"));
    }
}
