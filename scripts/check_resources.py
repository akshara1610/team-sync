"""
TeamSync - Resource Checker
Estimates resource requirements and checks current system
"""
import psutil
import platform
import subprocess
import sys


def check_gpu():
    """Check if GPU is available"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            gpu_info = result.stdout.strip()
            return True, gpu_info
        return False, "No NVIDIA GPU detected"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, "nvidia-smi not found (no GPU)"


def get_system_resources():
    """Get current system resources"""
    cpu_count = psutil.cpu_count(logical=False)
    cpu_count_logical = psutil.cpu_count(logical=True)
    ram_gb = psutil.virtual_memory().total / (1024**3)
    disk_gb = psutil.disk_usage('/').total / (1024**3)

    return {
        'cpu_physical': cpu_count,
        'cpu_logical': cpu_count_logical,
        'ram_gb': ram_gb,
        'disk_gb': disk_gb,
        'platform': platform.system(),
        'python_version': platform.python_version()
    }


def estimate_resource_usage():
    """Estimate resource usage for TeamSync components"""
    return {
        'components': {
            'Whisper (CPU)': {'ram': '2-4 GB', 'cpu': '2 cores'},
            'pyannote (CPU)': {'ram': '2-3 GB', 'cpu': '2 cores'},
            'ChromaDB': {'ram': '1-2 GB', 'cpu': '1 core', 'disk': '5-20 GB'},
            'PostgreSQL': {'ram': '512 MB', 'cpu': '1 core', 'disk': '1-5 GB'},
            'Redis': {'ram': '256 MB', 'cpu': '0.5 core'},
            'FastAPI': {'ram': '256 MB', 'cpu': '0.5 core'},
            'LangChain/MCP': {'ram': '512 MB', 'cpu': '0.5 core'},
        },
        'totals': {
            'minimum': {'ram': '8 GB', 'cpu': '4 cores', 'disk': '20 GB'},
            'recommended': {'ram': '16 GB', 'cpu': '8 cores', 'disk': '50 GB'},
            'with_gpu': {'ram': '16 GB', 'cpu': '8 cores', 'disk': '50 GB', 'gpu': 'NVIDIA T4 (12GB)'}
        }
    }


def check_requirements():
    """Check if current system meets requirements"""
    sys_resources = get_system_resources()
    estimates = estimate_resource_usage()

    min_req = estimates['totals']['minimum']
    rec_req = estimates['totals']['recommended']

    # Parse minimum requirements
    min_ram = float(min_req['ram'].split()[0])
    min_cpu = int(min_req['cpu'].split()[0])
    min_disk = float(min_req['disk'].split()[0])

    # Parse recommended requirements
    rec_ram = float(rec_req['ram'].split()[0])
    rec_cpu = int(rec_req['cpu'].split()[0])

    meets_minimum = (
        sys_resources['ram_gb'] >= min_ram and
        sys_resources['cpu_physical'] >= min_cpu and
        sys_resources['disk_gb'] >= min_disk
    )

    meets_recommended = (
        sys_resources['ram_gb'] >= rec_ram and
        sys_resources['cpu_physical'] >= rec_cpu and
        sys_resources['disk_gb'] >= min_disk
    )

    return meets_minimum, meets_recommended


def print_report():
    """Print comprehensive resource report"""
    print("=" * 70)
    print("TeamSync - System Resource Report")
    print("=" * 70)

    # Current system
    print("\n📊 CURRENT SYSTEM:")
    print("-" * 70)
    sys_res = get_system_resources()
    print(f"Platform:        {sys_res['platform']}")
    print(f"Python:          {sys_res['python_version']}")
    print(f"CPU Cores:       {sys_res['cpu_physical']} physical / {sys_res['cpu_logical']} logical")
    print(f"RAM:             {sys_res['ram_gb']:.1f} GB")
    print(f"Disk:            {sys_res['disk_gb']:.1f} GB")

    # GPU check
    has_gpu, gpu_info = check_gpu()
    print(f"GPU:             {'✓ ' + gpu_info if has_gpu else '✗ ' + gpu_info}")

    # Component estimates
    print("\n📦 ESTIMATED RESOURCE USAGE PER COMPONENT:")
    print("-" * 70)
    estimates = estimate_resource_usage()
    for component, usage in estimates['components'].items():
        ram = usage.get('ram', '-')
        cpu = usage.get('cpu', '-')
        disk = usage.get('disk', '-')
        print(f"{component:25} RAM: {ram:10}  CPU: {cpu:10}  Disk: {disk}")

    # Total requirements
    print("\n🎯 TOTAL REQUIREMENTS:")
    print("-" * 70)
    for tier, reqs in estimates['totals'].items():
        tier_name = tier.replace('_', ' ').title()
        print(f"\n{tier_name}:")
        for key, value in reqs.items():
            print(f"  {key.upper():10} {value}")

    # Recommendation
    print("\n💡 RECOMMENDATION:")
    print("-" * 70)
    meets_min, meets_rec = check_requirements()

    if not meets_min:
        print("❌ Current system does NOT meet minimum requirements")
        print("   → Upgrade needed for production use")
    elif meets_rec:
        print("✅ Current system meets recommended requirements")
        print("   → Ready for production use")
    else:
        print("⚠️  Current system meets minimum but not recommended requirements")
        print("   → Will work but may be slow under load")

    # GPU recommendation
    print("\n🖥️  GPU RECOMMENDATION:")
    print("-" * 70)
    if has_gpu:
        print("✓ GPU detected - transcription will be faster")
        print("  → No additional GPU needed")
    else:
        print("✗ No GPU detected - running in CPU mode")
        print("  → GPU NOT required, but transcription will be slower")
        print("  → For production: Consider GPU only if:")
        print("    • Real-time transcription is critical")
        print("    • Processing many meetings simultaneously")
        print("    • Cost of GPU (~$300/month extra) is acceptable")

    # GCP recommendations
    print("\n☁️  GCP MACHINE TYPE RECOMMENDATIONS:")
    print("-" * 70)
    print("\nOption 1 - CPU Only (Recommended):")
    print("  Machine Type:  n2-standard-4")
    print("  Specs:         4 vCPUs, 16 GB RAM")
    print("  Cost:          ~$150/month")
    print("  Best for:      Most use cases")

    print("\nOption 2 - With GPU (Optional):")
    print("  Machine Type:  n1-standard-4 + NVIDIA T4")
    print("  Specs:         4 vCPUs, 15 GB RAM, T4 GPU (16GB)")
    print("  Cost:          ~$450/month")
    print("  Best for:      Real-time transcription needs")

    print("\nOption 3 - High Performance:")
    print("  Machine Type:  n2-standard-8")
    print("  Specs:         8 vCPUs, 32 GB RAM")
    print("  Cost:          ~$300/month")
    print("  Best for:      High concurrent meeting load")

    # Cost optimization
    print("\n💰 COST OPTIMIZATION TIPS:")
    print("-" * 70)
    print("1. Use CPU-only - GPU not essential for this workload")
    print("2. All AI (GPT-4, embeddings) runs via OpenAI API (not local)")
    print("3. Only transcription/diarization run locally")
    print("4. For testing: Use local machine")
    print("5. For production: Start with n2-standard-4, scale if needed")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    try:
        print_report()
    except Exception as e:
        print(f"Error generating report: {e}")
        sys.exit(1)
