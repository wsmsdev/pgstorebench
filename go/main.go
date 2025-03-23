package main

import (
	"context"
	"fmt"
	"log"
	"math/rand"
	"os"
	"runtime"
	"sync/atomic"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/joho/godotenv"
	"golang.org/x/sync/errgroup"
)

//Define vector type
type Vector [1024]float32

func generateVectors(amount int, dimensions int) []Vector {
	start := time.Now()
	vectors := make([]Vector, amount)
	for i := range vectors {
		for j := range dimensions {
			vectors[i][j] = rand.Float32()
		}
	}
	elapsed := time.Since(start)
	fmt.Printf("Time taken to generate %d vectors: %s\n", amount, elapsed)
	return vectors
}

func main() {
	logger := log.New(os.Stdout, "", log.LstdFlags)
	start := time.Now()

	//*INFO Load .env from parent in this case
	err := godotenv.Load()
	if err != nil {
		err = godotenv.Load("../.env")
		if err != nil {
			logger.Fatalf("Error loading .env file: %v", err)
		}
	}
	dbUser := os.Getenv("POSTGRES_USER")
	dbPassword := os.Getenv("POSTGRES_PASSWORD")
	ctx := context.Background()
	
	poolConfig, err := pgxpool.ParseConfig(fmt.Sprintf("postgres://%s:%s@%s:%s/%s", dbUser, dbPassword, "localhost", "5432", "pgvector"))
	if err != nil {
		log.Fatalf("Unable to parse DB config: %v", err)
	}

	//TODO: Test different pool configurations
	poolConfig.MaxConns = 20
	poolConfig.MinConns = 8
	poolConfig.MaxConnLifetime = time.Hour
	poolConfig.MaxConnIdleTime = 30 * time.Second
	poolConfig.HealthCheckPeriod = time.Minute
	poolConfig.AfterConnect = func(ctx context.Context, conn *pgx.Conn) error {
		_, err := conn.Exec(ctx, "SET synchronous_commit = off")
		return err
	}
	
	//*INFO: This is the connection pool
	pool, err := pgxpool.NewWithConfig(ctx, poolConfig)
	if err != nil {
		log.Fatalf("Unable to create pool: %v", err)
	}
	defer pool.Close()

	// Create the table
	conn, err := pool.Acquire(ctx)
	if err != nil {
		log.Fatalf("Unable to acquire connection: %v", err)
	}
	_, err = conn.Exec(ctx, `
		DROP TABLE IF EXISTS vector_store;
		CREATE TABLE IF NOT EXISTS vector_store (
			id VARCHAR(255) PRIMARY KEY,
			embedding FLOAT[] NOT NULL
		)
	`)
	conn.Release()
	if err != nil {
		log.Fatalf("Unable to create table: %v", err)
	}

	// Generate vectors and insert them
	vectors := generateVectors(1000000, 1024)
	if err := insertVectors(vectors, pool, ctx, logger); err != nil {
		log.Fatalf("Error inserting vectors: %v", err)
	}

	elapsed := time.Since(start)
	fmt.Printf("Time taken to process all batches: %s\n", elapsed)
}

func insertVectors(vectors []Vector, pool *pgxpool.Pool, ctx context.Context, logger *log.Logger) error {
	start := time.Now()
	logger.Printf("Inserting vectors in batches")
	
	//* Define same batch size as in python
	batchSize := 800

	//* Create workers based on number of cores
	workers := runtime.NumCPU()
    
	//* Create channels for batches and results
	batchChan := make(chan []Vector, workers)
	resultChan := make(chan error, workers)

	//* Setup an error group for concurrent operations
	eg, ctx := errgroup.WithContext(ctx)

	//Phase 1 Batching
	eg.Go(func() error {
		defer close(batchChan)
		
		batch := make([]Vector, 0, batchSize)
		for _, vec := range vectors {
			batch = append(batch, vec)
			
			if len(batch) >= batchSize {
				select {
				case batchChan <- batch:
					batch = make([]Vector, 0, batchSize)
				case <-ctx.Done():
					return ctx.Err()
				}
			}
		}
		
		// Send the last batch if not empty
		if len(batch) > 0 {
			select {
			case batchChan <- batch:
			case <-ctx.Done():
				return ctx.Err()
			}
		}
		
		return nil
	})

	//* Atomic counter for Ids
	var batchCounter int64 = 0

	//Phase 2 Database Insertion
	for range make([]struct{}, workers) {
		eg.Go(func() error {
			for batch := range batchChan {
				offset := int(atomic.AddInt64(&batchCounter, 1) - 1) * batchSize
				if err := insertBatchWithCopy(ctx, pool, batch, offset); err != nil {
					return err
				}
			}
			return nil
		})
	}
		
	go func() {
		if err := eg.Wait(); err != nil {
			resultChan <- err
		}
		close(resultChan)
	}()
	
	for err := range resultChan {
		if err != nil {
			return err
		}
	}
	
	elapsed := time.Since(start)
	totalVectors := len(vectors)
	logger.Printf("Time taken to insert %d vectors: %s", totalVectors, elapsed)
	
	vectorsPerSecond := float64(totalVectors) / elapsed.Seconds()
	logger.Printf("Insertion rate: %.2f vectors/second", vectorsPerSecond)
	
	return nil
}

func insertBatchWithCopy(ctx context.Context, pool *pgxpool.Pool, batch []Vector, batchOffset int) error {
	conn, err := pool.Acquire(ctx)
	if err != nil {
		return fmt.Errorf("acquire connection: %w", err)
	}
	defer conn.Release()
	
	// Use COPY for most efficient bulk loading
	copyCount, err := conn.Conn().CopyFrom(
		ctx,
		pgx.Identifier{"vector_store"},
		[]string{"id", "embedding"},
		pgx.CopyFromSlice(len(batch), func(i int) ([]any, error) {
			return []any{
				fmt.Sprintf("id-%d", batchOffset + i), // Generate ID
				batch[i][:],          // Convert to slice
			}, nil
		}),
	)
	
	if err != nil {
		return fmt.Errorf("copy operation: %w", err)
	}
	
	if int(copyCount) != len(batch) {
		return fmt.Errorf("expected to copy %d rows, copied %d", len(batch), copyCount)
	}
	
	return nil
}
