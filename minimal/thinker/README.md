# TRANSFER THIS WHOLE DIRECTORY TO HPC
#
#   scp -r /Users/eb2007/playground/minimal/thinker eb2007@login.hpc.cam.ac.uk:~/rds/hpc-work/driver-thinker-practice/
#
# Then on an interactive GPU session (Wilkes3 / sintr), not the login node:
#
#   cd ~/rds/hpc-work/driver-thinker-practice/thinker
#   python -m venv .venv && source .venv/bin/activate
#   pip install -r requirements.txt
#   export HF_HOME=$HOME/rds/hpc-work/hf-cache
#   python server.py --host 127.0.0.1 --port 8000
#
# Laptop test (no GPU, no Qwen weights):
#   python server.py --mock --host 127.0.0.1 --port 8000
#
# Manual check:
#   curl -s http://127.0.0.1:8000/health
#   curl -s -F "image=@../data/frames/000000.jpg" -F "frame_index=0" http://127.0.0.1:8000/think
#
# Nothing else in this repo needs to run on HPC.
