which uv 2>/dev/null || python3 -m pip install uv 2>/dev/null || pip install uv
which nvcc 2>/dev/null || module load cuda 2>/dev/null || echo "请手动加载CUDA环境"
test -d ~/lab11 || echo "请先上传lab11到~/lab11"
cd ~/lab11
mkdir -p bin results results/figures
rm -rf bin/*
chmod +x scripts/build.sh scripts/run_conv.sh
./scripts/build.sh
./scripts/build.sh cudnn
./scripts/run_conv.sh 32 1 1
./scripts/run_conv.sh 64 1 1 8 8
./scripts/run_conv.sh 32 1 2
./scripts/run_conv.sh 64 2 2
./scripts/run_conv.sh 32 1 3
./scripts/run_conv.sh 64 3 3
./scripts/run_conv.sh 32 1 4
uv run python3 -c "import matplotlib; import numpy" 2>/dev/null || uv pip install matplotlib numpy
uv run python scripts/benchmark.py
uv run python scripts/plot.py
ls results/figures/ | head
cd ~
tar -czf lab11_results.tar.gz lab11/results/ lab11/report/
ls -lh lab11_results.tar.gz
