from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Any
import threading
from config import CONFIG, logger


def process_parallel(
    items: List[Any],
    process_func: Callable,
    max_workers: int = None,
    batch_num: int = None,
    total_batches: int = None
) -> List[Any]:
    """
    Process items in parallel using a thread pool.

    Args:
        items: List of items to process
        process_func: Function to apply to each item (returns result or None)
        max_workers: Number of concurrent workers (default: CONFIG["max_workers"])
        batch_num: Current batch number for logging
        total_batches: Total number of batches for logging

    Returns:
        List of (item, result) tuples for successful results
    """
    max_workers = max_workers or CONFIG["max_workers"]
    results = []
    total_items = len(items)
    completed_count = 0
    count_lock = threading.Lock()

    batch_prefix = f"Batch {batch_num}/{total_batches}" if batch_num else "Processing"

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(process_func, item): item for item in items}

        for future in as_completed(future_to_item):
            item = future_to_item[future]
            with count_lock:
                completed_count += 1
                current = completed_count

            try:
                result = future.result()
                if result is not None:
                    results.append((item, result))
                    logger.info(f"{batch_prefix}, {current}/{total_items}, Name: {item}")
                else:
                    logger.warning(f"{batch_prefix}, {current}/{total_items}, Name: {item} - FAILED")
            except Exception as e:
                logger.error(f"{batch_prefix}, {current}/{total_items}, Name: {item} - ERROR: {e}")

    return results
