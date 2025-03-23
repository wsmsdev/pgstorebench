import asyncio
import asyncpg
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import io
import struct
from dotenv import load_dotenv
import os
import time
import logging

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

USER = os.getenv("POSTGRES_USER")
PASSWORD = os.getenv("POSTGRES_PASSWORD")
DATABASE = os.getenv("POSTGRES_DB")
HOST = "localhost"
PORT = 5432


def generate_synth_vector(dimensions: int, amount: int) -> np.ndarray:
    logger.info(f"Generating {amount} vectors with {dimensions} dimensions")
    start = time.time()
    vectors = np.random.randn(amount, dimensions).astype(np.float32)
    end = time.time()
    logger.info(f"Time taken to generate {amount} vectors: {end - start} seconds")
    return vectors


async def async_pg_connect() -> asyncpg.Pool:
    logger.info("Connecting to database")
    return await asyncpg.create_pool(
        f"postgres://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}",
        min_size=8,
        max_size=20,
        command_timeout=60,
        statement_cache_size=0,
        max_cached_statement_lifetime=0,
        max_inactive_connection_lifetime=30,
    )


async def create_table(conn: asyncpg.Connection):
    logger.info("Creating table")
    await conn.execute(
        """
    CREATE TABLE IF NOT EXISTS vector_table (
        id SERIAL PRIMARY KEY,
        vector_id VARCHAR(255) NOT NULL,
        vector_data FLOAT[] NOT NULL
    )
    """
    )


async def remove_table(conn: asyncpg.Connection):
    logger.info("Removing table")
    await conn.execute("DROP TABLE IF EXISTS vector_table")


async def insert_in_batches(
    pool: asyncpg.Connection,
    vectors: np.ndarray,
    batch_size: int = 800,
):
    logger.info("Inserting vectors in batches")
    start = time.time()

    # Create batched vector groups based on batch size and amount of vectors
    batches = [vectors[i : i + batch_size] for i in range(0, len(vectors), batch_size)]

    # Init Semaphore to control the number of concurrent connections
    sem = asyncio.Semaphore(8)

    async def process_batch(batch: np.ndarray):
        async with sem:
            async with pool.acquire() as conn:
                await conn.execute("SET statement_timeout = 60000")
                # Convert each vector to a proper PostgreSQL array format
                records = [
                    (str(i), list(v))  # vector_id as string, vector_data as list
                    for i, v in enumerate(batch)
                ]

                await conn.copy_records_to_table(
                    "vector_table",
                    records=records,
                    columns=("vector_id", "vector_data"),
                )

    logger.info(f"Starting to process {len(batches)} batches")
    await asyncio.gather(*[process_batch(batch) for batch in batches])

    end = time.time()
    logger.info(f"Time taken to insert {len(vectors)} vectors: {end - start} seconds")


async def main(max_workers: int, pool: asyncpg.Connection, vector_amount: int):
    start = time.time()
    logger.info(f"Starting main with {max_workers} workers")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        vectors = generate_synth_vector(1024, vector_amount)
        chunk_size = len(vectors) // max_workers
        chunks = [
            vectors[i : i + chunk_size] for i in range(0, len(vectors), chunk_size)
        ]

        await asyncio.gather(*[insert_in_batches(pool, chunk) for chunk in chunks])
    end = time.time()
    logger.info(f"Time taken to process all batches: {end - start} seconds")


if __name__ == "__main__":

    async def run():
        pool = await async_pg_connect()
        await remove_table(pool)
        await create_table(pool)
        await main(8, pool, 100000)
        await pool.close()

    asyncio.run(run())
