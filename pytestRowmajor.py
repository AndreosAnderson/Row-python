import numpy as np
import pytest
import time
import psutil
from memory_profiler import memory_usage

def track_memory(func, *args):
    mem_usage = memory_usage((func, args))
    print(f"Memory usage: {max(mem_usage)} MB")
    return mem_usage


def track_cpu(func, *args):
    cpu_percent_before = psutil.cpu_percent(interval=1)
    start_time = time.time()

    func(*args)

    end_time = time.time()
    cpu_percent_after = psutil.cpu_percent(interval=1)

    print(f"Initial CPU usage: {cpu_percent_before}%")
    print(f"Final CPU usage: {cpu_percent_after}%")
    print(f"Execution time: {end_time - start_time} seconds")

def matrix_multiply(A,B,C,N):
    if  A.shape[1] == B.shape[0]:
        for i in range(0,N,1): 
            for j in range(0,N,1):
                for k in range(0,N,1):
                  C[i,j]+= A[i,k]*B[k,j]
        return C
    else:
        return "Sorry, cannot multiply A and B."

@pytest.fixture
def setup_matrices():
    size = 1024
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    C = np.zeros((A.shape[0],B.shape[1]),dtype = float)
    return A, B, C, size

@pytest.mark.benchmark(min_rounds=5)
def test_matrix_multiply(benchmark, setup_matrices):
    A, B , C, N = setup_matrices

    result = benchmark(matrix_multiply, A, B, C, N)
    track_memory(matrix_multiply, A, B, C, N)
    track_cpu(matrix_multiply, A, B, C, N)
    assert result is not None
