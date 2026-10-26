package com.hirematch.repository;

import com.hirematch.entity.Job;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.Optional;

@Repository
public interface JobRepository extends JpaRepository<Job, Long> {
    
    Optional<Job> findByHash(String hash);

    @Transactional
    @Modifying
    @Query(value = "UPDATE jobs SET embedding = CAST(:vectorString AS vector), updated_at = NOW() WHERE id = :id", nativeQuery = true)
    void updateEmbedding(Long id, String vectorString);

    @Query(value = "SELECT (j.embedding IS NOT NULL) FROM jobs j WHERE j.id = :id", nativeQuery = true)
    Boolean hasEmbedding(Long id);

    /**
     * Pairwise semantic textual similarity between a candidate resume and a specific job
     * using pgvector cosine distance (<=>). Cosine similarity = 1 - cosine_distance.
     * Excludes jobs without active status or missing embeddings.
     */
    @Query(value = "SELECT GREATEST(0.0, LEAST(1.0, (1 - (j.embedding <=> r.embedding)))) " +
                   "FROM jobs j, resumes r " +
                   "WHERE r.id = :resumeId AND j.id = :jobId " +
                   "  AND j.is_active = true " +
                   "  AND j.embedding IS NOT NULL AND r.embedding IS NOT NULL", nativeQuery = true)
    Optional<Double> findCosineSimilarity(Long resumeId, Long jobId);

    /**
     * Finds top-N recommended jobs ranked by genuine semantic similarity (ascending cosine distance).
     * Strictly filters for active jobs and valid non-null embeddings on both sides.
     */
    @Query(value = "SELECT j.id as job_id, GREATEST(0.0, LEAST(1.0, (1 - (j.embedding <=> r.embedding)))) as similarity " +
                   "FROM jobs j, resumes r " +
                   "WHERE r.id = :resumeId " +
                   "  AND j.is_active = true " +
                   "  AND j.embedding IS NOT NULL AND r.embedding IS NOT NULL " +
                   "ORDER BY j.embedding <=> r.embedding ASC " +
                   "LIMIT :limit", nativeQuery = true)
    List<Object[]> findRecommendedJobIdsWithScore(Long resumeId, int limit);

    /**
     * Semantic job recommendations with optional workMode and jobType filters and pagination.
     */
    @Query(value = "SELECT j.id as job_id, GREATEST(0.0, LEAST(1.0, (1 - (j.embedding <=> r.embedding)))) as similarity " +
                   "FROM jobs j, resumes r " +
                   "WHERE r.id = :resumeId " +
                   "  AND j.is_active = true " +
                   "  AND j.embedding IS NOT NULL AND r.embedding IS NOT NULL " +
                   "  AND (:workMode IS NULL OR :workMode = '' OR :workMode = 'All' OR UPPER(j.work_mode) = UPPER(:workMode)) " +
                   "  AND (:jobType IS NULL OR :jobType = '' OR :jobType = 'All' OR UPPER(j.job_type) = UPPER(:jobType)) " +
                   "ORDER BY j.embedding <=> r.embedding ASC " +
                   "LIMIT :limit OFFSET :offset", nativeQuery = true)
    List<Object[]> findRecommendedJobIdsWithFilters(Long resumeId, String workMode, String jobType, int limit, int offset);

    @Query(value = "SELECT COUNT(j.id) FROM jobs j, resumes r " +
                   "WHERE r.id = :resumeId " +
                   "  AND j.is_active = true " +
                   "  AND j.embedding IS NOT NULL AND r.embedding IS NOT NULL " +
                   "  AND (:workMode IS NULL OR :workMode = '' OR :workMode = 'All' OR UPPER(j.work_mode) = UPPER(:workMode)) " +
                   "  AND (:jobType IS NULL OR :jobType = '' OR :jobType = 'All' OR UPPER(j.job_type) = UPPER(:jobType))", nativeQuery = true)
    long countRecommendedJobs(Long resumeId, String workMode, String jobType);

    List<Job> findTop20ByIsActiveTrueOrderByPostedAtDesc();

    @Query("SELECT j FROM Job j WHERE " +
           "(:query IS NULL OR :query = '' OR LOWER(j.title) LIKE LOWER(CONCAT('%', :query, '%')) OR LOWER(j.company) LIKE LOWER(CONCAT('%', :query, '%')) OR LOWER(j.location) LIKE LOWER(CONCAT('%', :query, '%'))) AND " +
           "(:workMode IS NULL OR :workMode = 'All' OR :workMode = '' OR UPPER(j.workMode) = UPPER(:workMode)) AND " +
           "(:jobType IS NULL OR :jobType = 'All' OR :jobType = '' OR UPPER(j.jobType) = UPPER(:jobType)) " +
           "ORDER BY j.postedAt DESC NULLS LAST, j.createdAt DESC")
    Page<Job> searchJobs(String query, String workMode, String jobType, Pageable pageable);
}
