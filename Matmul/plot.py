import subprocess
import time
import matplotlib.pyplot as plt

# Compile all programs
def compile_programs():
    cpp_program = 'cpu_matmul'
    programs = ['cublas_matmul', 'naive_matmul', 'opt_matmul']
    subprocess.run(['g++', f'{cpp_program}.cpp', '-o', cpp_program], check=True)
    for program in programs:
        if program == 'cublas_matmul':
            subprocess.run(['nvcc', f'{program}.cu', '-o', program, '-lcublas'], check=True)
        else:
            subprocess.run(['nvcc', f'{program}.cu', '-o', program], check=True)

sizes = [256, 512, 1024, 2048, 4096, 8192]

def run_program(program):
    times_per_size = []
    for size in sizes:
        total_time = 0.0
        for _ in range(3):
            start = time.perf_counter()
            subprocess.run([f'./{program}', str(size)], check=True)
            end = time.perf_counter()
            total_time += end - start
        times_per_size.append(total_time / 3)
    return times_per_size

def plot_results(results):
    plt.figure(figsize=(10, 6))
    for program, times in results.items():
        plt.plot(sizes, times, label=program)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Matrix Size (N x N)')
    plt.ylabel('Time (seconds)')
    plt.title('Matrix Multiplication Performance')
    plt.legend()
    plt.grid(True)
    plt.savefig('performance_plot.png')
    plt.show()

def main():
    compile_programs()
    results = {}
    for program in ['cublas_matmul', 'naive_matmul', 'opt_matmul']:
        results[program] = run_program(program)
    plot_results(results)

if __name__ == "__main__":
    main()
